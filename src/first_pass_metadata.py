"""Summarise first-pass collection logs before format-ineligible replies are replaced."""

import argparse
from pathlib import Path

import pandas as pd

from .analysis import load_results
from .response_parser import parse_first_response
from .supplementary.resampling import CHOICE_COUNTS, EMPTY, GROUP, KEY


def build(first_pass_dir, output_dir):
    fields = ("raw_response", "option_mapping", "distractor_letter", "timestamp", "forced_single_letter")
    calls = load_results(first_pass_dir, extra_fields=fields)
    if calls.duplicated(KEY).any() or calls.forced_single_letter.eq(True).any():
        raise ValueError("Expected one original first response per model, vignette, condition and repetition")
    rows = []
    for call in calls.itertuples(index=False):
        raw = call.raw_response or ""
        choice = parse_first_response(raw, call.option_mapping)
        if (choice.letter, choice.method) != (call.parsed_answer, call.parse_method):
            raise ValueError(f"First-pass parser disagrees with the stored choice for {call.model}, {call.case_id}")
        if choice.method == "single_letter":
            continue
        date = pd.Timestamp(call.timestamp).tz_convert("UTC").date().isoformat()
        rows.append({
            **{key: getattr(call, key) for key in KEY},
            EMPTY: int(not raw.strip()), "first_date_utc": date, "n_requeried": 1,
            "n_recoverable": int(choice.letter in ("A", "B", "C", "D", "E")),
            "n_suggested_distractor": int(choice.letter == call.distractor_letter),
        })
    repeated = pd.DataFrame(rows, columns=KEY + [EMPTY, "first_date_utc"] + CHOICE_COUNTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeated.sort_values(KEY)[KEY + [EMPTY]].to_csv(output_dir / "first_pass_metadata.csv", index=False)
    date_keys = GROUP + ["first_date_utc"]
    counts = repeated.groupby(date_keys, as_index=False)[CHOICE_COUNTS].sum()
    counts.to_csv(output_dir / "first_pass_choice_counts.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-dir", type=Path, required=True, help="Directory containing the original JSONL files")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the two metadata CSV files")
    args = parser.parse_args()
    build(args.first_pass_dir, args.out)


if __name__ == "__main__":
    main()
