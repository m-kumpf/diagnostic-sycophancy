"""Held-out neutral baselines for cases answered correctly in all five selection draws."""

import pandas as pd

from .common import (BASELINE_REPS, MODEL_NAME, SEED, SELECT_REPS, TRIGGERS, case_counts, load_calls, ordered_models,
                        pct, rate_ci, rate_diff, save_table, split_half_calls)


def describe(model, definition, sub, baseline_label):
    cases, cnt = case_counts(sub)
    row = {"model": MODEL_NAME.get(model, model), "definition": definition, "n_cases": len(cases),
           "neutral_baseline": baseline_label}
    for j, cond in enumerate(("neutral", "mild", "severe")):
        for outcome, name in (("is_correct", "accuracy"), ("chose_distractor", "distractor")):
            est, lo, hi = rate_ci(cnt, cond, outcome, SEED + 410 + j)
            row[f"{name}_{cond}_pct"], row[f"{name}_{cond}_ci_lo"], row[f"{name}_{cond}_ci_hi"] = pct(est), pct(lo), pct(hi)
    for trig in TRIGGERS:
        est, lo, hi = rate_diff(cnt, "chose_distractor", "neutral", trig, SEED + 420)
        row[f"distractor_lift_{trig}_pp"], row[f"distractor_lift_{trig}_ci_lo"], row[f"distractor_lift_{trig}_ci_hi"] = pct(est), pct(lo), pct(hi)
        est, lo, hi = rate_diff(cnt, "is_correct", trig, "neutral", SEED + 440)
        row[f"accuracy_drop_{trig}_pp"], row[f"accuracy_drop_{trig}_ci_lo"], row[f"accuracy_drop_{trig}_ci_hi"] = pct(est), pct(lo), pct(hi)
    return row


def main():
    df = load_calls()
    rows = []
    for model in ordered_models(df):
        dm = df[df["model"] == model]
        rows.append(describe(model, "split-half: select on neutral reps 1-5, baseline on reps 6-10",
                             split_half_calls(dm), "held-out neutral reps 6-10"))
        rows.append(describe(model, "split-half reversed: select on neutral reps 6-10, baseline on reps 1-5",
                             split_half_calls(dm, BASELINE_REPS, SELECT_REPS), "held-out neutral reps 1-5"))
        # For the all-ten subset, selection and baseline use the same repetitions.
        all_ten = tuple(range(1, 11))
        rows.append(describe(model, "all ten: 10/10 neutral reps correct (baseline fixed by selection)",
                             split_half_calls(dm, all_ten, all_ten), "same draws as selection"))
    table = pd.DataFrame(rows)
    save_table(table, "high_consistency.csv")


if __name__ == "__main__":
    main()
