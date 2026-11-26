"""First-pass response format, distractor selection and requested model settings."""

import pandas as pd

from ..config import MODELS
from .common import CONDITIONS, METADATA_DIR, MODEL_NAME, MODEL_ORDER, OUT_DIR, load_calls, save_table

KEY = ["model", "case_id", "condition", "repetition"]
EMPTY = "first_response_empty"
GROUP = ["model", "condition"]
CHOICE_COUNTS = ["n_requeried", "n_recoverable", "n_suggested_distractor"]
OPEN_WEIGHTS = {"deepseek-v4-pro", "glm-5.1"}


def validate_design(final):
    if final.duplicated(KEY).any():
        raise ValueError("Duplicate slot in analysed JSONL")
    expected_reps = set(range(1, 11))
    cells = final.groupby(KEY[:-1], observed=True)["repetition"].agg(set)
    cases = final["case_id"].unique()
    expected = pd.MultiIndex.from_product([MODEL_ORDER, cases, CONDITIONS], names=KEY[:-1])
    if (len(cases) != 100 or len(cells.index.symmetric_difference(expected)) or
            not cells.map(lambda reps: reps == expected_reps).all()):
        raise ValueError("Expected five models, the same 100 cases and ten repetitions per condition")


def load_metadata(final, path):
    metadata = pd.read_csv(path)
    if list(metadata.columns) != KEY + [EMPTY] or metadata.isna().any().any():
        raise ValueError("Unexpected metadata columns or missing data")
    if not metadata[EMPTY].isin([0, 1]).all():
        raise ValueError("First-response empty flag must be 0 or 1")
    flagged = final.loc[final["requeried"], KEY]
    check = flagged.merge(metadata, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not check["_merge"].eq("both").all():
        raise ValueError("Metadata keys must match all re-queried slots exactly")
    full = final.merge(metadata, on=KEY, how="left", validate="one_to_one")
    full[EMPTY] = full[EMPTY].fillna(0).astype(int)
    return full


def summarise(records, model, condition):
    n = len(records)
    requery = int(records["requeried"].sum())
    empty = int(records[EMPTY].sum())
    return {"model": model, "condition": condition, "n_cells": n,
            "n_first_response_single_letter": n - requery, "n_requeried": requery,
            "pct_requeried": round(100 * requery / n, 1), "n_first_response_empty": empty,
            "pct_first_response_empty": round(100 * empty / n, 1)}


def load_choice_counts(calls):
    counts = pd.read_csv(METADATA_DIR / "first_pass_choice_counts.csv")
    keys = GROUP + ["first_date_utc"]
    if list(counts.columns) != keys + CHOICE_COUNTS or counts.isna().any().any() or counts.duplicated(keys).any():
        raise ValueError("Invalid first-pass choice count keys or columns")
    for col in CHOICE_COUNTS:
        if not pd.api.types.is_integer_dtype(counts[col]) or counts[col].lt(0).any():
            raise ValueError(f"Counts must be nonnegative integers: {col}")
    if (counts.n_requeried.eq(0).any() or counts.n_recoverable.gt(counts.n_requeried).any()
            or counts.n_suggested_distractor.gt(counts.n_recoverable).any()):
        raise ValueError("Inconsistent first-pass choice totals")
    pd.to_datetime(counts.first_date_utc, format="%Y-%m-%d", utc=True, errors="raise")
    totals = counts.groupby(GROUP).sum(numeric_only=True)
    flagged = calls[calls.requeried].groupby(GROUP).agg(n=("case_id", "size"), empty_count=(EMPTY, "sum"))
    if not totals.n_requeried.equals(flagged.n) or totals.n_recoverable.gt(flagged.n - flagged.empty_count).any():
        raise ValueError("First-pass counts disagree with the recorded repeats and empty responses")
    return counts


def format_table(calls):
    rows = []
    for model in MODEL_ORDER:
        group = calls[calls.model == model]
        for condition in CONDITIONS:
            rows.append(summarise(group[group.condition == condition], MODEL_NAME[model], condition))
        rows.append(summarise(group, MODEL_NAME[model], "all conditions"))
    for condition in CONDITIONS:
        rows.append(summarise(calls[calls.condition == condition], "All models", condition))
    rows.append(summarise(calls, "All models", "all conditions"))
    save_table(pd.DataFrame(rows), "format_resampling.csv")


def first_pass_comparison(calls, counts):
    rows = []
    for (model, condition), group in calls.groupby(GROUP):
        unrepeated = group[~group.requeried]
        repeated = counts[(counts.model == model) & (counts.condition == condition)]
        first_n = int(unrepeated.parseable.sum() + repeated.n_recoverable.sum())
        first_d = int(unrepeated.chose_distractor.sum() + repeated.n_suggested_distractor.sum())
        final_n = int(group.parseable.sum())
        final_d = int(group.chose_distractor.sum())
        rows.append({
            "model": MODEL_NAME[model], "condition": condition,
            "n_first_recoverable": first_n, "n_first_suggested_distractor": first_d,
            "first_distractor_pct": 100 * first_d / first_n,
            "n_final_parseable": final_n, "n_final_suggested_distractor": final_d,
            "final_distractor_pct": 100 * final_d / final_n,
        })
    rates = pd.DataFrame(rows)
    lifts = []
    for model, group in rates.groupby("model"):
        by_cond = group.set_index("condition")
        first = by_cond.loc["severe", "first_distractor_pct"] - by_cond.loc["neutral", "first_distractor_pct"]
        final = by_cond.loc["severe", "final_distractor_pct"] - by_cond.loc["neutral", "final_distractor_pct"]
        lifts.append({"model": model, "first_pass_lift_pp": first, "final_lift_pp": final,
                      "final_minus_first_pp": final - first})
    save_table(rates.round(6), "first_pass_rates.csv")
    save_table(pd.DataFrame(lifts).round(6), "first_pass_lifts.csv")


def study_materials(calls, counts):
    rows = []
    for model, group in calls.groupby("model"):
        model_id = MODELS[model].model_id
        if set(group.model_id) != {model_id} or set(group.temperature) != {1.0}:
            raise ValueError(f"Unexpected endpoint or temperature for {model}")
        dates = pd.to_datetime(group.timestamp, utc=True).dt.strftime("%Y-%m-%d")
        initial = set(dates[~group.requeried]) | set(counts.loc[counts.model == model, "first_date_utc"])
        repeated = sorted(set(dates[group.requeried]))
        rows.append({"model": MODEL_NAME[model], "requested_model_id": model_id,
                     "model_weights": "Open" if model in OPEN_WEIGHTS else "Proprietary",
                     "first_dates_utc": ", ".join(sorted(initial)), "repeat_dates_utc": ", ".join(repeated),
                     "n_requeried": int(group.requeried.sum()), "n_slots": len(group),
                     "temperature_requested": 1.0})
    save_table(pd.DataFrame(rows), "request_settings.csv")
    example = calls[(calls.model == "gemini-3.1-pro-preview") & (calls.case_id == "ORTHO_02")
                    & (calls.repetition == 1)].set_index("condition")
    prompts = [f"{condition.title()}\n\n{example.loc[condition, 'prompt']}" for condition in CONDITIONS]
    (OUT_DIR / "example_prompts.txt").write_text("\n\n".join(prompts) + "\n", encoding="utf-8")


def main():
    calls = load_calls(extra=("forced_single_letter", "model_id", "temperature", "timestamp", "prompt"))
    calls["condition"] = calls["condition"].astype(object)
    validate_design(calls)
    flags = calls.forced_single_letter.dropna()
    if not flags.map(lambda flag: isinstance(flag, bool)).all():
        raise ValueError("Expected boolean repeat flags")
    calls["requeried"] = calls.forced_single_letter.eq(True)
    calls = load_metadata(calls, METADATA_DIR / "first_pass_metadata.csv")
    counts = load_choice_counts(calls)
    format_table(calls)
    first_pass_comparison(calls, counts)
    study_materials(calls, counts)


if __name__ == "__main__":
    main()
