"""Convert a per-model JSONL output file to a flat CSV for inspection.

Usage:
    python -m src.export_csv results/raw/gemini-3.1-pro-preview.pilot.jsonl
    python -m src.export_csv results/raw/gemini-3.1-pro-preview.jsonl --out summary.csv --include-prompt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data_loader import SPECIALTY_MAP


CORE_COLUMNS = [
    "call_id",
    "model",
    "case_id",
    "specialty",
    "condition",
    "repetition",
    "parsed_answer",
    "parse_method",
    "chosen_diagnosis",
    "correct_letter",
    "correct_diagnosis",
    "distractor_letter",
    "distractor_diagnosis",
    "is_correct",
    "chose_distractor",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "option_E",
    "raw_response",
    "error",
    "attempts",
    "latency_ms",
    "finish_reason",
    "timestamp",
    "random_seed",
    "model_id",
    "temperature",
]


def _specialty(case_id: str) -> str:
    prefix = case_id.split("_", 1)[0]
    return SPECIALTY_MAP.get(prefix, "")


def _flatten(rec: dict, include_prompt: bool) -> dict:
    option_mapping = rec.get("option_mapping") or {}
    parsed = rec.get("parsed_answer")
    chosen_diagnosis = option_mapping.get(parsed, "") if parsed in option_mapping else ""

    correct_letter = rec.get("correct_letter")
    distractor_letter = rec.get("distractor_letter")

    row = {
        "call_id": rec.get("call_id", ""),
        "model": rec.get("model", ""),
        "case_id": rec.get("case_id", ""),
        "specialty": _specialty(rec.get("case_id", "")),
        "condition": rec.get("condition", ""),
        "repetition": rec.get("repetition", ""),
        "parsed_answer": parsed or "",
        "parse_method": rec.get("parse_method", ""),
        "chosen_diagnosis": chosen_diagnosis,
        "correct_letter": correct_letter or "",
        "correct_diagnosis": option_mapping.get(correct_letter, "") if correct_letter else "",
        "distractor_letter": distractor_letter or "",
        "distractor_diagnosis": option_mapping.get(distractor_letter, "") if distractor_letter else "",
        "is_correct": rec.get("is_correct", ""),
        "chose_distractor": rec.get("chose_distractor", ""),
        "option_A": option_mapping.get("A", ""),
        "option_B": option_mapping.get("B", ""),
        "option_C": option_mapping.get("C", ""),
        "option_D": option_mapping.get("D", ""),
        "option_E": option_mapping.get("E", ""),
        "raw_response": rec.get("raw_response") or "",
        "error": rec.get("error") or "",
        "attempts": rec.get("attempts", ""),
        "latency_ms": rec.get("latency_ms", ""),
        "finish_reason": rec.get("finish_reason") or "",
        "timestamp": rec.get("timestamp", ""),
        "random_seed": rec.get("random_seed", ""),
        "model_id": rec.get("model_id", ""),
        "temperature": rec.get("temperature", ""),
    }
    if include_prompt:
        row["prompt"] = rec.get("prompt", "")
    return row


def convert(input_path: Path, output_path: Path, include_prompt: bool) -> int:
    columns = list(CORE_COLUMNS)
    if include_prompt:
        columns.append("prompt")

    n = 0
    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warning: skipping bad line: {e}")
                continue
            writer.writerow(_flatten(rec, include_prompt))
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a JSONL output file to CSV.")
    parser.add_argument("input", type=Path, help="Path to a .jsonl file produced by run_experiment.")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path. Defaults to the input path with .csv suffix.",
    )
    parser.add_argument(
        "--include-prompt", action="store_true",
        help="Include the full prompt column (bulky; off by default).",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    output_path = args.out if args.out is not None else args.input.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = convert(args.input, output_path, args.include_prompt)
    print(f"Wrote {n} rows -> {output_path}")


if __name__ == "__main__":
    main()
