"""Model-specific Table 2 contrasts and sensitivity to a family spanning five models."""

import pandas as pd

from ..analysis import _holm
from .common import ANALYSIS_DIR, MODEL_NAME, save_table

CONTRASTS = [
    ("dd_a", "mild-neutral", "Distractor lift, mild vs neutral"),
    ("dd_a", "severe-neutral", "Distractor lift, severe vs neutral (primary)"),
    ("dd_a", "severe-mild", "Distractor contrast, severe vs mild"),
    ("accuracy_drop", "neutral-mild", "Accuracy drop, mild vs neutral"),
    ("accuracy_drop", "neutral-severe", "Accuracy drop, severe vs neutral"),
]


def main():
    primary = pd.read_csv(ANALYSIS_DIR / "inference_primary.csv", float_precision="round_trip").set_index("model")
    secondary = pd.read_csv(ANALYSIS_DIR / "inference_secondary.csv", float_precision="round_trip")
    secondary = secondary.set_index(["model", "outcome", "contrast"])
    rows = []
    for model, primary_test in primary.iterrows():
        for outcome, contrast, label in CONTRASTS:
            is_primary = (outcome, contrast) == ("dd_a", "severe-neutral")
            test = primary_test if is_primary else secondary.loc[(model, outcome, contrast)]
            rows.append({
                "model": MODEL_NAME[model], "contrast": label,
                "risk_difference_pp": 100 * test["rd_lift" if is_primary else "rd"],
                "rd_ci_lo_pp": 100 * test.rd_lo, "rd_ci_hi_pp": 100 * test.rd_hi,
                "odds_ratio": test.OR, "or_ci_lo": test.OR_lo, "or_ci_hi": test.OR_hi,
                "p_value": test["p_gee" if is_primary else "p_holm"],
                "p_adjustment": "unadjusted" if is_primary else "Holm within model",
            })
    pd.DataFrame(rows).to_csv(ANALYSIS_DIR / "table_2.csv", index=False)
    sensitivity = primary[["p_gee", "p_perm"]].copy()
    sensitivity["gee_holm_across_models"] = _holm(sensitivity.p_gee.to_list())
    sensitivity["permutation_holm_across_models"] = _holm(sensitivity.p_perm.to_list())
    sensitivity.index = sensitivity.index.map(MODEL_NAME)
    save_table(sensitivity.reset_index(), "primary_holm_sensitivity.csv")


if __name__ == "__main__":
    main()
