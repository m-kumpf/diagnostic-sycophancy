"""Accuracy and distractor selection in orthopaedic and other vignettes."""

import numpy as np
import pandas as pd

from .common import (MODEL_NAME, SEED, TRIGGERS, boot_ci, case_counts, load_calls, ordered_models,
                        pct, perm_p, rate, rate_diff, save_table)

ORTHO = "orthopedics"


def stratum_row(model, label, d):
    cases, cnt = case_counts(d)
    row = {"model": MODEL_NAME.get(model, model), "stratum": label, "n_cases": len(cases)}
    for cond in ("neutral", "mild", "severe"):
        row[f"accuracy_{cond}_pct"] = pct(rate(cnt, cond, "is_correct"))
        row[f"distractor_{cond}_pct"] = pct(rate(cnt, cond, "chose_distractor"))
    for trig in TRIGGERS:
        primary = trig == "severe"
        est, lo, hi = rate_diff(cnt, "chose_distractor", "neutral", trig, SEED + (100 if primary else 300))
        row[f"distractor_lift_{trig}_pp"] = pct(est)
        row[f"distractor_lift_{trig}_ci_lo"] = pct(lo)
        row[f"distractor_lift_{trig}_ci_hi"] = pct(hi)
        row[f"distractor_lift_{trig}_p_perm"] = perm_p(cnt, "chose_distractor", "neutral", trig,
                                                        SEED + (200 if primary else 400))
        est, lo, hi = rate_diff(cnt, "is_correct", trig, "neutral", SEED + 300)
        row[f"accuracy_drop_{trig}_pp"] = pct(est)
        row[f"accuracy_drop_{trig}_ci_lo"] = pct(lo)
        row[f"accuracy_drop_{trig}_ci_hi"] = pct(hi)
    return row


def difference_rows(model, d):
    """Distractor lift in orthopaedics minus lift in the other specialties; cases resampled across both strata."""
    cases, cnt = case_counts(d)
    is_ortho = np.array([c.startswith("ORTHO") for c in cases])
    rows = []
    for trig in TRIGGERS:
        def lift(sample, mask):
            sel = sample[mask[sample]]
            if len(sel) == 0:
                return np.nan
            return rate(cnt, trig, "chose_distractor", sel) - rate(cnt, "neutral", "chose_distractor", sel)

        stat = lambda s: lift(s, is_ortho) - lift(s, ~is_ortho)
        full = np.arange(len(cases))
        lo, hi = boot_ci(stat, len(cases), SEED + 500)
        rows.append({
            "model": MODEL_NAME.get(model, model), "trigger": trig,
            "lift_orthopaedics_pp": pct(lift(full, is_ortho)),
            "lift_other_specialties_pp": pct(lift(full, ~is_ortho)),
            "difference_pp": pct(stat(full)), "difference_ci_lo": pct(lo), "difference_ci_hi": pct(hi),
        })
    return rows


def main():
    df = load_calls()
    strata, diffs = [], []
    for model in ordered_models(df):
        dm = df[df["model"] == model]
        strata.append(stratum_row(model, "all cases", dm))
        strata.append(stratum_row(model, "orthopaedics", dm[dm["specialty"] == ORTHO]))
        strata.append(stratum_row(model, "other specialties", dm[dm["specialty"] != ORTHO]))
        diffs.extend(difference_rows(model, dm))
    strata, diffs = pd.DataFrame(strata), pd.DataFrame(diffs)
    save_table(strata, "specialty_strata.csv")
    save_table(diffs, "specialty_difference.csv")


if __name__ == "__main__":
    main()
