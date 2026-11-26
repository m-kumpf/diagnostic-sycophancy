"""Accuracy loss, changes in wrong-option selection and the distribution of case-level effects."""

import numpy as np
import pandas as pd

from .common import (MODEL_NAME, SEED, TRIGGERS, case_counts, load_calls, ordered_models, pct, per_case_rate,
                        rate_diff, save_table)

OUTCOMES = ("is_correct", "chose_distractor", "other_wrong")


def top_share(pos_sorted, k):
    """Share of the summed positive lift that comes from the k cases with the largest lift."""
    total = pos_sorted.sum()
    return round(100 * pos_sorted[:k].sum() / total, 1) if total > 0 else None


def main():
    df = load_calls()
    decomposition, summary, per_case = [], [], []
    for model in ordered_models(df):
        name = MODEL_NAME.get(model, model)
        cases, cnt = case_counts(df[df["model"] == model], outcomes=OUTCOMES)
        rates = {(cond, o): per_case_rate(cnt, cond, o) for cond in ("neutral", "mild", "severe") for o in OUTCOMES}
        for trig in TRIGGERS:
            drop = rate_diff(cnt, "is_correct", trig, "neutral", SEED + 300)
            sugg = rate_diff(cnt, "chose_distractor", "neutral", trig, SEED + (100 if trig == "severe" else 300))
            other = rate_diff(cnt, "other_wrong", "neutral", trig, SEED + 600)
            decomposition.append({
                "model": name, "trigger": trig,
                "accuracy_drop_pp": pct(drop[0]), "accuracy_drop_ci_lo": pct(drop[1]), "accuracy_drop_ci_hi": pct(drop[2]),
                "to_suggested_distractor_pp": pct(sugg[0]), "to_suggested_ci_lo": pct(sugg[1]), "to_suggested_ci_hi": pct(sugg[2]),
                "to_other_wrong_options_pp": pct(other[0]), "to_other_ci_lo": pct(other[1]), "to_other_ci_hi": pct(other[2]),
            })
            lift = rates[(trig, "chose_distractor")] - rates[("neutral", "chose_distractor")]
            pos = np.sort(lift[lift > 0])[::-1]
            cum = np.cumsum(pos) / pos.sum() if pos.sum() > 0 else np.array([])
            summary.append({
                "model": name, "trigger": trig, "n_cases": len(lift),
                "mean_lift_pp": pct(lift.mean()),
                "n_cases_lift_positive": int((lift > 0).sum()),
                "n_cases_lift_zero": int((lift == 0).sum()),
                "n_cases_lift_negative": int((lift < 0).sum()),
                "n_cases_lift_at_least_50pp": int((lift >= 0.5).sum()),
                "largest_case_lift_pp": pct(lift.max()),
                "share_of_positive_lift_from_top5_cases_pct": top_share(pos, 5),
                "share_of_positive_lift_from_top10_cases_pct": top_share(pos, 10),
                "n_cases_for_half_of_positive_lift": int(np.searchsorted(cum, 0.5 - 1e-12) + 1) if len(cum) else None,
            })
        for i, case in enumerate(cases):
            per_case.append({
                "model": name, "case_id": case,
                "accuracy_neutral_pct": pct(rates[("neutral", "is_correct")][i]),
                "accuracy_mild_pct": pct(rates[("mild", "is_correct")][i]),
                "accuracy_severe_pct": pct(rates[("severe", "is_correct")][i]),
                "distractor_neutral_pct": pct(rates[("neutral", "chose_distractor")][i]),
                "distractor_mild_pct": pct(rates[("mild", "chose_distractor")][i]),
                "distractor_severe_pct": pct(rates[("severe", "chose_distractor")][i]),
                "lift_mild_pp": pct(rates[("mild", "chose_distractor")][i] - rates[("neutral", "chose_distractor")][i]),
                "lift_severe_pp": pct(rates[("severe", "chose_distractor")][i] - rates[("neutral", "chose_distractor")][i]),
            })
    decomposition, summary = pd.DataFrame(decomposition), pd.DataFrame(summary)
    summary["n_cases_for_half_of_positive_lift"] = summary["n_cases_for_half_of_positive_lift"].astype("Int64")
    save_table(decomposition, "accuracy_decomposition.csv")
    save_table(summary, "case_change_summary.csv")
    save_table(pd.DataFrame(per_case), "case_changes.csv")


if __name__ == "__main__":
    main()
