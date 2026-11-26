"""Case-level pressure effects by neutral distractor selection in the four other models."""

import numpy as np
import pandas as pd

from .common import (MODEL_NAME, N_PERM, SEED, TRIGGERS, boot_ci, case_counts, load_calls, ordered_models,
                        pct, per_case_rate, rate, rate_diff, save_table, subset_counts)


def rank_test(x, y, seed):
    """Two-sided label permutation using midranks; integer products count ties exactly."""
    rx = (2 * pd.Series(x).rank()).to_numpy(dtype=np.int64) - (len(x) + 1)
    ry = (2 * pd.Series(y).rank()).to_numpy(dtype=np.int64) - (len(y) + 1)
    denominator = np.sqrt(float(rx @ rx) * float(ry @ ry))
    if denominator == 0:
        return np.nan, np.nan
    observed = int(rx @ ry)
    rng = np.random.default_rng(seed)
    hits = sum(abs(int(rng.permutation(rx) @ ry)) >= abs(observed) for _ in range(N_PERM))
    return observed / denominator, (hits + 1) / (N_PERM + 1)


def main():
    df = load_calls()
    models = ordered_models(df)
    counts = {m: case_counts(df[df["model"] == m]) for m in models}
    cases = counts[models[0]][0]
    for m in models:
        model_cases, model_counts = counts[m]
        if not np.array_equal(model_cases, cases) or any((model_counts[c]["n"] != 10).any() for c in ("neutral", *TRIGGERS)):
            raise ValueError("Plausibility ranks require the same cases and ten retained draws per condition in each model")
    neutral = {m: per_case_rate(counts[m][1], "neutral", "chose_distractor") for m in models}
    pooled = np.mean([neutral[m] for m in models], axis=0)

    summary, case_rows = [], []
    for k, model in enumerate(models):
        cnt = counts[model][1]
        other_counts = np.sum([counts[m][1]["neutral"]["chose_distractor"].astype(int)
                               for m in models if m != model], axis=0)
        index = other_counts / (10 * (len(models) - 1))
        never, ever = index == 0, index > 0
        for trig in TRIGGERS:
            lift_counts = cnt[trig]["chose_distractor"].astype(int) - cnt["neutral"]["chose_distractor"].astype(int)
            rho, rank_p = rank_test(other_counts, lift_counts, SEED + 320 + k)
            est_n, lo_n, hi_n = rate_diff(subset_counts(cnt, never), "chose_distractor", "neutral", trig, SEED + 310)
            est_e, lo_e, hi_e = rate_diff(subset_counts(cnt, ever), "chose_distractor", "neutral", trig, SEED + 311)

            def group_lift(sample, mask):
                sel = sample[mask[sample]]
                if len(sel) == 0:
                    return np.nan
                return rate(cnt, trig, "chose_distractor", sel) - rate(cnt, "neutral", "chose_distractor", sel)

            diff = lambda s: group_lift(s, ever) - group_lift(s, never)
            d_lo, d_hi = boot_ci(diff, len(cases), SEED + 312)
            summary.append({
                "model": MODEL_NAME.get(model, model), "trigger": trig,
                "n_cases_never_chosen": int(never.sum()),
                "lift_never_chosen_pp": pct(est_n), "lift_never_ci_lo": pct(lo_n), "lift_never_ci_hi": pct(hi_n),
                "n_cases_chosen_at_least_once": int(ever.sum()),
                "lift_chosen_pp": pct(est_e), "lift_chosen_ci_lo": pct(lo_e), "lift_chosen_ci_hi": pct(hi_e),
                "difference_pp": pct(diff(np.arange(len(cases)))), "difference_ci_lo": pct(d_lo), "difference_ci_hi": pct(d_hi),
                "spearman_rho": round(rho, 3),
                "spearman_p_perm": rank_p,
            })
        for i, case in enumerate(cases):
            case_rows.append({
                "model": MODEL_NAME.get(model, model), "case_id": case,
                "neutral_selection_all_models_pct": pct(pooled[i]),
                "neutral_selection_other_models_pct": pct(index[i]),
                "neutral_selection_this_model_pct": pct(neutral[model][i]),
                "lift_mild_pp": pct(per_case_rate(cnt, "mild", "chose_distractor")[i] - neutral[model][i]),
                "lift_severe_pp": pct(per_case_rate(cnt, "severe", "chose_distractor")[i] - neutral[model][i]),
            })
    summary = pd.DataFrame(summary)
    save_table(summary, "distractor_plausibility.csv")
    save_table(pd.DataFrame(case_rows), "distractor_plausibility_cases.csv")


if __name__ == "__main__":
    main()
