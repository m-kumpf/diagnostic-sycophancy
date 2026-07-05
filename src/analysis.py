"""Descriptive analysis for the sycophancy experiment.

Computes, per model and condition: accuracy, DD_a (sycophancy-distractor) rate,
the DD_a lift over the neutral baseline, accuracy drop, resistance, and
case-level susceptibility — all with cluster-bootstrap-by-case 95% CIs.

Design decisions:
  * Sycophancy headline = DD_a *lift over neutral*; raw rates reported too.
  * Primary endpoint = severe-neutral DD_a lift, per model (two-sided a=0.05).
  * CIs via cluster bootstrap resampling CASES (not calls), since reps within a
    case are correlated.
  * Unparseable calls are excluded from accuracy/DD_a denominators (reported
    separately); they never enter a numerator (is_correct/chose_distractor are
    False for them).

Forward-compatible: auto-discovers every results/raw/*.jsonl (more models or
cases just drop in); nothing about the model list, case count, condition set,
or rep count is hardcoded.

Usage:
    python -m src.analysis [--results-dir results/raw] [--out results/analysis] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data_loader import SPECIALTY_MAP


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CONDITIONS = ["neutral", "mild", "severe"]
TRIGGERS = ["mild", "severe"]
SEED = 20260524
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _specialty(case_id: str) -> str:
    return SPECIALTY_MAP.get(case_id.split("_", 1)[0], "unknown")


def load_results(results_dir: Path) -> pd.DataFrame:
    """One row per call across all (non-pilot) model files in results_dir."""
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        if "pilot" in path.name or path.name.endswith(".tmp"):
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append({
                    "model": r["model"],
                    "case_id": r["case_id"],
                    "specialty": _specialty(r["case_id"]),
                    "condition": r["condition"],
                    "repetition": r["repetition"],
                    "parsed_answer": r["parsed_answer"],
                    "parse_method": r.get("parse_method", ""),
                    "is_correct": bool(r["is_correct"]),
                    "chose_distractor": bool(r["chose_distractor"]),
                    "parseable": r["parsed_answer"] != "unparseable",
                })
    if not rows:
        raise SystemExit(f"No result files found in {results_dir}")
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(
        df["condition"],
        categories=[c for c in CANONICAL_CONDITIONS if c in df["condition"].unique()],
        ordered=True,
    )
    return df


# --------------------------------------------------------------------------- #
# Per-model per-case arrays (drive fast cluster bootstrap)
# --------------------------------------------------------------------------- #
def _model_arrays(df_model: pd.DataFrame):
    """Aligned per-case count arrays for each condition: tot/parseable/correct/dd_a."""
    cases = sorted(df_model["case_id"].unique())
    pos = {c: i for i, c in enumerate(cases)}
    conds = [c for c in CANONICAL_CONDITIONS if c in df_model["condition"].unique()]
    arr = {}
    for cond in conds:
        sub = df_model[df_model["condition"] == cond]
        n = len(cases)
        tot, par, cor, dda = (np.zeros(n) for _ in range(4))
        for case, g in sub.groupby("case_id", observed=True):
            i = pos[case]
            tot[i] = len(g)
            par[i] = g["parseable"].sum()
            cor[i] = g["is_correct"].sum()
            dda[i] = g["chose_distractor"].sum()
        arr[cond] = {"tot": tot, "par": par, "cor": cor, "dda": dda}
    return cases, conds, arr


def _rate(a: dict, num_key: str, sample: np.ndarray) -> float:
    num = a[num_key][sample].sum()
    den = a["par"][sample].sum()
    return num / den if den > 0 else np.nan


def _bootstrap_ci(n_cases: int, stat_fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.integers(0, n_cases, n_cases)
        vals[b] = stat_fn(sample)
    lo, hi = np.nanpercentile(vals, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return lo, hi


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def rates_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    out = []
    for model, dfm in df.groupby("model"):
        cases, conds, arr = _model_arrays(dfm)
        n = len(cases)
        full = np.arange(n)
        for cond in conds:
            a = arr[cond]
            acc = _rate(a, "cor", full)
            dda = _rate(a, "dda", full)
            acc_ci = _bootstrap_ci(n, lambda s, a=a: _rate(a, "cor", s), n_boot, SEED)
            dda_ci = _bootstrap_ci(n, lambda s, a=a: _rate(a, "dda", s), n_boot, SEED + 1)
            out.append({
                "model": model, "condition": cond,
                "n_calls": int(a["tot"].sum()), "n_parseable": int(a["par"].sum()),
                "unparseable_rate": 1 - a["par"].sum() / a["tot"].sum(),
                "accuracy": acc, "accuracy_lo": acc_ci[0], "accuracy_hi": acc_ci[1],
                "dd_a_rate": dda, "dd_a_lo": dda_ci[0], "dd_a_hi": dda_ci[1],
            })
    return pd.DataFrame(out)


def lifts_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """DD_a lift (trigger - neutral) and accuracy drop (neutral - trigger), per model."""
    out = []
    for model, dfm in df.groupby("model"):
        cases, conds, arr = _model_arrays(dfm)
        if "neutral" not in conds:
            continue
        n = len(cases)
        full = np.arange(n)

        def add(outcome, num_key, cond_a, cond_b, primary):
            if cond_a not in conds or cond_b not in conds:
                return
            aa, ab = arr[cond_a], arr[cond_b]
            stat = lambda s: _rate(ab, num_key, s) - _rate(aa, num_key, s)
            est = stat(full)
            lo, hi = _bootstrap_ci(n, stat, n_boot, SEED + 7)
            out.append({
                "model": model, "outcome": outcome,
                "contrast": f"{cond_b}_minus_{cond_a}",
                "estimate": est, "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": (lo > 0) or (hi < 0),
                "is_primary": primary,
            })

        # DD_a lift (trigger relative to neutral, and severe relative to mild)
        add("dd_a", "dda", "neutral", "mild", primary=False)
        add("dd_a", "dda", "neutral", "severe", primary=True)   # PRIMARY ENDPOINT
        add("dd_a", "dda", "mild", "severe", primary=False)
        # Accuracy drop (neutral relative to trigger -> positive = accuracy fell)
        for trig in TRIGGERS:
            if trig in conds:
                an, at = arr["neutral"], arr[trig]
                stat = lambda s, at=at: _rate(an, "cor", s) - _rate(at, "cor", s)
                est = stat(full)
                lo, hi = _bootstrap_ci(n, stat, n_boot, SEED + 9)
                out.append({
                    "model": model, "outcome": "accuracy_drop",
                    "contrast": f"neutral_minus_{trig}",
                    "estimate": est, "ci_lo": lo, "ci_hi": hi,
                    "excludes_zero": (lo > 0) or (hi < 0), "is_primary": False,
                })
    return pd.DataFrame(out)


def case_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, case): rate of correct / DD_a in each condition, plus lifts."""
    g = (df.groupby(["model", "case_id", "specialty", "condition"], observed=True)
           .agg(n=("parseable", "size"), par=("parseable", "sum"),
                cor=("is_correct", "sum"), dda=("chose_distractor", "sum"))
           .reset_index())
    g["acc"] = np.where(g["par"] > 0, g["cor"] / g["par"], np.nan)
    g["dda_rate"] = np.where(g["par"] > 0, g["dda"] / g["par"], np.nan)
    acc = g.pivot_table(index=["model", "case_id", "specialty"], columns="condition",
                        values="acc", observed=True)
    dda = g.pivot_table(index=["model", "case_id", "specialty"], columns="condition",
                        values="dda_rate", observed=True)
    acc.columns = [f"acc_{c}" for c in acc.columns]
    dda.columns = [f"dda_{c}" for c in dda.columns]
    case = acc.join(dda).reset_index()
    if {"dda_severe", "dda_neutral"}.issubset(case.columns):
        case["dda_lift_severe"] = case["dda_severe"] - case["dda_neutral"]
    if {"dda_mild", "dda_neutral"}.issubset(case.columns):
        case["dda_lift_mild"] = case["dda_mild"] - case["dda_neutral"]
    for trig in TRIGGERS:
        if {f"acc_{trig}", "acc_neutral"}.issubset(case.columns):
            case[f"acc_drop_{trig}"] = case["acc_neutral"] - case[f"acc_{trig}"]
    return case


def heterogeneity_table(case: pd.DataFrame) -> pd.DataFrame:
    """Per model: spread of per-case neutral DD_a and severe-neutral lift; difficulty corr."""
    out = []
    for model, c in case.groupby("model"):
        row = {"model": model}
        for col, label in [("dda_neutral", "neutral_dd_a"), ("dda_lift_severe", "severe_lift")]:
            if col in c.columns:
                v = c[col].dropna()
                row[f"{label}_sd"] = v.std()
                row[f"{label}_iqr"] = v.quantile(0.75) - v.quantile(0.25)
                row[f"{label}_min"] = v.min()
                row[f"{label}_max"] = v.max()
        # Does case difficulty (neutral accuracy) predict susceptibility (severe lift)?
        if {"acc_neutral", "dda_lift_severe"}.issubset(c.columns):
            d = c[["acc_neutral", "dda_lift_severe"]].dropna()
            if len(d) > 2 and d["acc_neutral"].nunique() > 1:
                rho, p = stats.spearmanr(d["acc_neutral"], d["dda_lift_severe"])
                row["spearman_difficulty_vs_lift"] = rho
                row["spearman_p"] = p
        out.append(row)
    return pd.DataFrame(out)


def resistance_table(df: pd.DataFrame, case: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Continuous accuracy drop (primary) + perfect-neutral-accuracy subset (secondary)."""
    out = []
    for model, dfm in df.groupby("model"):
        c = case[case["model"] == model]
        cases, conds, arr = _model_arrays(dfm)
        n = len(cases)
        for trig in TRIGGERS:
            drop_col = f"acc_drop_{trig}"
            if drop_col not in c.columns:
                continue
            mean_drop = c[drop_col].mean()
            # bootstrap the mean per-case drop over cases
            an, at = arr["neutral"], arr[trig]
            per_case = (np.where(an["par"] > 0, an["cor"] / an["par"], np.nan)
                        - np.where(at["par"] > 0, at["cor"] / at["par"], np.nan))
            stat = lambda s: np.nanmean(per_case[s])
            lo, hi = _bootstrap_ci(n, stat, n_boot, SEED + 11)
            # secondary: cases with perfect neutral accuracy
            perfect = c[c["acc_neutral"] == 1.0]
            # On the perfect-neutral subset, dda_neutral is 0 by construction
            # (every neutral rep was correct, so no rep picked DD_a).
            assert (perfect["dda_neutral"] == 0).all() if len(perfect) else True
            pn_dd_a_shift = perfect[f"dda_{trig}"].mean() if len(perfect) else np.nan
            pn_acc_drop = (1.0 - perfect[f"acc_{trig}"].mean()) if len(perfect) else np.nan
            n_perf = len(perfect)
            if n_perf >= 2:
                dda_vals = perfect[f"dda_{trig}"].to_numpy()
                acc_vals = perfect[f"acc_{trig}"].to_numpy()
                pn_dd_lo, pn_dd_hi = _bootstrap_ci(
                    n_perf, lambda s, v=dda_vals: v[s].mean(), n_boot, SEED + 13
                )
                pn_acc_lo, pn_acc_hi = _bootstrap_ci(
                    n_perf, lambda s, v=acc_vals: 1.0 - v[s].mean(), n_boot, SEED + 15
                )
            else:
                pn_dd_lo = pn_dd_hi = pn_acc_lo = pn_acc_hi = np.nan
            row = {
                "model": model, "trigger": trig,
                "mean_accuracy_drop": mean_drop, "drop_lo": lo, "drop_hi": hi,
                "n_perfect_neutral": int(n_perf),
                "perfect_neutral_trigger_acc": perfect[f"acc_{trig}"].mean() if n_perf else np.nan,
                "pn_dd_a_shift": pn_dd_a_shift,
                "pn_dd_a_shift_lo": pn_dd_lo, "pn_dd_a_shift_hi": pn_dd_hi,
                "pn_acc_drop": pn_acc_drop,
                "pn_acc_drop_lo": pn_acc_lo, "pn_acc_drop_hi": pn_acc_hi,
            }
            out.append(row)
    return pd.DataFrame(out)


def lift_by_difficulty_table(case: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per-model mean DD_a lift (mild and severe vs neutral) split by whether
    the model answered the case correctly in every neutral repetition.

    Two subsets per model:
      - `mixed`: cases with 0 <= acc_neutral < 1 (model wasn't always right in neutral)
      - `perfect`: cases with acc_neutral == 1 (model was always right in neutral)
    """
    out = []
    for model, c in case.groupby("model"):
        for subset_name, subset in [("mixed", c[c["acc_neutral"] < 1.0]),
                                    ("perfect", c[c["acc_neutral"] == 1.0])]:
            n = len(subset)
            for contrast, col in [("mild-neutral", "dda_lift_mild"),
                                  ("severe-neutral", "dda_lift_severe")]:
                if col not in subset.columns:
                    continue
                vals = subset[col].dropna().to_numpy()
                if len(vals) >= 2:
                    mean_lift = float(vals.mean())
                    lo, hi = _bootstrap_ci(len(vals),
                                           lambda s, v=vals: v[s].mean(),
                                           n_boot, SEED + 17)
                else:
                    mean_lift = float(vals.mean()) if len(vals) else np.nan
                    lo = hi = np.nan
                out.append({
                    "model": model, "subset": subset_name, "contrast": contrast,
                    "n_cases": int(n),
                    "mean_lift": mean_lift, "lift_lo": lo, "lift_hi": hi,
                })
    return pd.DataFrame(out)


def perfect_neutral_inference_table(case: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per-model condition contrasts of the DD_a-selection rate restricted to the
    perfect-neutral subset (cases each model answered correctly in every neutral
    repetition). Three contrasts per model:
      - severe-neutral: mean dda_severe on subset (since dda_neutral = 0 by construction)
      - mild-neutral:   mean dda_mild on subset
      - severe-mild:    mean(dda_severe - dda_mild) on subset
    Cluster bootstrap by case; two-sided empirical p; Holm-corrected within model."""
    rows = []
    for i, (model, c_all) in enumerate(case.groupby("model")):
        sub = c_all[c_all["acc_neutral"] == 1.0].dropna(
            subset=["dda_mild", "dda_severe"]).reset_index(drop=True)
        n = len(sub)
        if n < 4:
            continue
        dda_mild = sub["dda_mild"].to_numpy()
        dda_severe = sub["dda_severe"].to_numpy()
        contrasts = {
            "severe-neutral": dda_severe,
            "mild-neutral":   dda_mild,
            "severe-mild":    dda_severe - dda_mild,
        }
        model_rows = []
        for ci, (contrast, vals) in enumerate(contrasts.items()):
            obs = float(vals.mean())
            rng = np.random.default_rng(SEED + 23 + i * 7 + ci)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                boots[b] = vals[idx].mean()
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            under_null = boots - obs
            p_two = float(np.mean(np.abs(under_null) >= abs(obs)))
            p_two = max(p_two, 1.0 / n_boot)
            model_rows.append({
                "model": model, "contrast": contrast, "n_cases": int(n),
                "mean_lift": obs, "lift_lo": float(lo), "lift_hi": float(hi),
                "p_bootstrap": p_two,
            })
        p_holm_vals = _holm([r["p_bootstrap"] for r in model_rows])
        for r, ph in zip(model_rows, p_holm_vals):
            r["p_holm"] = float(ph)
        rows.extend(model_rows)
    return pd.DataFrame(rows)


def subset_contrast_table(case: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per-model test of the within-model gap
    mean_lift_severe[mixed] - mean_lift_severe[perfect],
    with cluster-bootstrap CI and two-sided empirical p-value (Holm across models)."""
    rows = []
    for i, (model, c) in enumerate(case.groupby("model")):
        c = c.dropna(subset=["acc_neutral", "dda_lift_severe"]).reset_index(drop=True)
        n = len(c)
        if n < 4:
            continue
        mixed_mask = (c["acc_neutral"] < 1.0).to_numpy()
        perfect_mask = (c["acc_neutral"] == 1.0).to_numpy()
        lifts = c["dda_lift_severe"].to_numpy()

        def _gap(idx):
            mm = mixed_mask[idx]; pm = perfect_mask[idx]
            if mm.sum() == 0 or pm.sum() == 0:
                return np.nan
            return lifts[idx][mm].mean() - lifts[idx][pm].mean()

        g_obs = _gap(np.arange(n))
        rng = np.random.default_rng(SEED + 19 + i)
        g_boots = np.empty(n_boot)
        for b in range(n_boot):
            g_boots[b] = _gap(rng.integers(0, n, n))
        gap_lo, gap_hi = np.nanpercentile(g_boots, [2.5, 97.5])
        # Two-sided empirical p: shift bootstrap distribution to null (mean 0)
        # and count two-sided exceedances of the observed gap. Floor at 1/n_boot.
        under_null = g_boots - g_obs
        p_two = float(np.mean(np.abs(under_null) >= abs(g_obs)))
        p_two = max(p_two, 1.0 / n_boot)
        rows.append({
            "model": model, "contrast": "severe-neutral",
            "gap_mixed_minus_perfect": float(g_obs),
            "gap_lo": float(gap_lo), "gap_hi": float(gap_hi),
            "p_bootstrap": p_two,
        })
    if rows:
        p_holm_vals = _holm([r["p_bootstrap"] for r in rows])
        for r, ph in zip(rows, p_holm_vals):
            r["p_holm"] = float(ph)
    return pd.DataFrame(rows)


def specialty_table(df: pd.DataFrame) -> pd.DataFrame:
    """Secondary, CAVEATED: DD_a rate & lift by specialty x condition (imbalanced!)."""
    g = (df[df["parseable"]]
         .groupby(["model", "specialty", "condition"], observed=True)
         .agg(n_cases=("case_id", "nunique"), n_parseable=("parseable", "size"),
              accuracy=("is_correct", "mean"), dd_a_rate=("chose_distractor", "mean"))
         .reset_index())
    return g


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Descriptive analysis.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "raw")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "analysis")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)

    df = load_results(args.results_dir)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["model"].unique())
    print(f"Loaded {len(df)} calls | models: {models} | "
          f"cases: {df['case_id'].nunique()} | conditions: {list(df['condition'].cat.categories)} "
          f"| bootstrap={args.n_boot}")

    rates = rates_table(df, args.n_boot)
    lifts = lifts_table(df, args.n_boot)
    case = case_table(df)
    heterog = heterogeneity_table(case)
    resist = resistance_table(df, case, args.n_boot)
    lift_diff = lift_by_difficulty_table(case, args.n_boot)
    subset_contrast = subset_contrast_table(case, args.n_boot)
    pn_inf = perfect_neutral_inference_table(case, args.n_boot)
    specialty = specialty_table(df)

    suscept = (case[["model", "case_id", "specialty", "dda_neutral", "dda_severe", "dda_lift_severe"]]
               .sort_values(["model", "dda_lift_severe"], ascending=[True, False])
               if "dda_lift_severe" in case.columns else case)

    rates.to_csv(out_dir / "descriptive_rates.csv", index=False)
    lifts.to_csv(out_dir / "lifts.csv", index=False)
    case.to_csv(out_dir / "case_level.csv", index=False)
    heterog.to_csv(out_dir / "case_heterogeneity.csv", index=False)
    suscept.to_csv(out_dir / "susceptibility_ranking.csv", index=False)
    resist.to_csv(out_dir / "resistance.csv", index=False)
    lift_diff.to_csv(out_dir / "dd_a_lift_by_difficulty.csv", index=False)
    subset_contrast.to_csv(out_dir / "dd_a_subset_contrast.csv", index=False)
    pn_inf.to_csv(out_dir / "dd_a_perfect_neutral_inference.csv", index=False)
    specialty.to_csv(out_dir / "specialty_breakdown.csv", index=False)

    # ---- printed summary ----
    pd.set_option("display.width", 140, "display.max_columns", 20)
    print("\n=== Rates (accuracy, DD_a) with 95% cluster-bootstrap CI ===")
    show = rates.copy()
    for col in ["accuracy", "accuracy_lo", "accuracy_hi", "dd_a_rate", "dd_a_lo", "dd_a_hi", "unparseable_rate"]:
        show[col] = (show[col] * 100).round(1)
    print(show[["model", "condition", "n_parseable", "unparseable_rate",
                "accuracy", "accuracy_lo", "accuracy_hi",
                "dd_a_rate", "dd_a_lo", "dd_a_hi"]].to_string(index=False))

    print("\n=== DD_a lift / accuracy drop (Δ proportion, 95% CI; * = primary endpoint) ===")
    pl = lifts.copy()
    for col in ["estimate", "ci_lo", "ci_hi"]:
        pl[col] = (pl[col] * 100).round(1)
    pl["sig"] = np.where(pl["excludes_zero"], "*CI excl 0*", "")
    pl["prim"] = np.where(pl["is_primary"], "PRIMARY", "")
    print(pl[["model", "outcome", "contrast", "estimate", "ci_lo", "ci_hi", "sig", "prim"]].to_string(index=False))

    print("\n=== Resistance (mean per-case accuracy drop; perfect-neutral subset) ===")
    pr = resist.copy()
    pct_cols = [
        "mean_accuracy_drop", "drop_lo", "drop_hi",
        "perfect_neutral_trigger_acc",
        "pn_dd_a_shift", "pn_dd_a_shift_lo", "pn_dd_a_shift_hi",
        "pn_acc_drop", "pn_acc_drop_lo", "pn_acc_drop_hi",
    ]
    for col in pct_cols:
        if col in pr.columns:
            pr[col] = (pr[col] * 100).round(1)
    print(pr.to_string(index=False))

    print("\n=== Case heterogeneity & difficulty→susceptibility (Spearman) ===")
    print(heterog.round(3).to_string(index=False))

    print(f"\nWrote CSVs to {out_dir}")
    print("Figures: run `python -m src.figures` (renders from these CSVs).")
    print("NOTE: specialty_breakdown.csv is descriptive only — specialties are imbalanced "
          "(ORTHO 47 … PED 4); not a powered comparison.")


if __name__ == "__main__":
    main()
