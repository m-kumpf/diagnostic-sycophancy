"""Case-clustered binomial GEE contrasts, bootstrap risk differences and paired sign-flip tests."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .analysis import (PROJECT_ROOT, SEED, _bootstrap_ci, _holm, _model_arrays, _rate,
                       load_results)

ALPHA = 0.05


def _fit_gee(d: pd.DataFrame, outcome: str, cov=None):
    """Fit a per-(sub)group GEE; returns (res, converged_flag)."""
    dd = d.copy()
    dd["y"] = dd[outcome].astype(int)
    dd["condition"] = dd["condition"].astype(str)
    cov = cov if cov is not None else Exchangeable()
    formula = "y ~ C(condition, Treatment(reference='neutral'))"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = smf.gee(formula, "case_id", data=dd,
                        family=sm.families.Binomial(), cov_struct=cov)
        res = model.fit()
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return res, converged


def _term_index(names, token):
    for i, n in enumerate(names):
        if token in n:
            return i
    raise KeyError(f"{token} not in {names}")


def _contrast(res, plus=None, minus=None):
    """Return (OR, ci_lo_OR, ci_hi_OR, p) for a +plus/-minus log-odds contrast."""
    names = res.model.exog_names
    vec = np.zeros(len(names))
    if plus:
        vec[_term_index(names, plus)] += 1.0
    if minus:
        vec[_term_index(names, minus)] -= 1.0
    tt = res.t_test(vec)
    est = float(np.ravel(tt.effect)[0])
    p = float(np.ravel(tt.pvalue)[0])
    lo, hi = np.ravel(tt.conf_int())[:2]
    return np.exp(est), np.exp(lo), np.exp(hi), p


def _per_case_rates(arr, cond, num_key):
    a = arr[cond]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a["par"] > 0, a[num_key] / a["par"], np.nan)


def _rd_lift(arr, cond_a, cond_b, num_key, n_boot, seed):
    """Risk-difference lift (cond_b - cond_a) with cluster-bootstrap CI."""
    n = len(arr[cond_a]["par"])
    full = np.arange(n)
    stat = lambda s: _rate(arr[cond_b], num_key, s) - _rate(arr[cond_a], num_key, s)
    est = stat(full)
    lo, hi = _bootstrap_ci(n, stat, n_boot, seed)
    return est, lo, hi


def _permutation_p(arr, cond_a, cond_b, num_key, n_perm, seed):
    """Sign-flip permutation test on the mean per-case lift (uses all paired cases)."""
    a = _per_case_rates(arr, cond_a, num_key)
    b = _per_case_rates(arr, cond_b, num_key)
    d = (b - a)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm_means = np.abs((signs * d).mean(axis=1))
    return (np.sum(perm_means >= obs) + 1) / (n_perm + 1)


def _wilcoxon_p(arr, cond_a, cond_b, num_key):
    a = _per_case_rates(arr, cond_a, num_key)
    b = _per_case_rates(arr, cond_b, num_key)
    d = (b - a)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz) < 6:
        return np.nan, int(len(nz))
    return float(stats.wilcoxon(nz, method="approx").pvalue), int(len(nz))


def primary_table(df, n_boot, n_perm):
    rows = []
    for model, dfm in df.groupby("model"):
        _, conds, arr = _model_arrays(dfm)
        if not {"neutral", "severe"}.issubset(conds):
            continue
        res, conv = _fit_gee(dfm, "chose_distractor")
        or_, lo, hi, p_gee = _contrast(res, plus="T.severe")
        rd, rd_lo, rd_hi = _rd_lift(arr, "neutral", "severe", "dda", n_boot, SEED + 100)
        p_perm = _permutation_p(arr, "neutral", "severe", "dda", n_perm, SEED + 200)
        p_wil, nz = _wilcoxon_p(arr, "neutral", "severe", "dda")
        rows.append({
            "model": model, "endpoint": "severe-neutral DD_a (PRIMARY)",
            "OR": or_, "OR_lo": lo, "OR_hi": hi, "p_gee": p_gee,
            "rd_lift": rd, "rd_lo": rd_lo, "rd_hi": rd_hi, "rd_excludes_0": (rd_lo > 0) or (rd_hi < 0),
            "p_perm": p_perm, "p_wilcoxon": p_wil, "wilcoxon_n_nonzero": nz,
            "gee_converged": conv,
        })
    return pd.DataFrame(rows)


def secondary_table(df, n_boot, n_perm):
    rows = []
    for model, dfm in df.groupby("model"):
        _, conds, arr = _model_arrays(dfm)
        if "neutral" not in conds:
            continue
        res_dda, conv_d = _fit_gee(dfm, "chose_distractor")
        res_acc, conv_a = _fit_gee(dfm, "is_correct")

        specs = []  # (outcome, contrast_label, res, plus, minus, num_key, cond_a, cond_b)
        if "mild" in conds:
            specs.append(("dd_a", "mild-neutral", res_dda, "T.mild", None, "dda", "neutral", "mild"))
        if {"mild", "severe"}.issubset(conds):
            specs.append(("dd_a", "severe-mild", res_dda, "T.severe", "T.mild", "dda", "mild", "severe"))
        if "mild" in conds:
            specs.append(("accuracy_drop", "neutral-mild", res_acc, "T.mild", None, "cor", "mild", "neutral"))
        if "severe" in conds:
            specs.append(("accuracy_drop", "neutral-severe", res_acc, "T.severe", None, "cor", "severe", "neutral"))

        fam = []
        for outcome, label, res, plus, minus, num_key, ca, cb in specs:
            or_, lo, hi, p = _contrast(res, plus=plus, minus=minus)
            rd, rd_lo, rd_hi = _rd_lift(arr, ca, cb, num_key, n_boot, SEED + 300)
            p_perm = _permutation_p(arr, ca, cb, num_key, n_perm, SEED + 400)
            fam.append({
                "model": model, "outcome": outcome, "contrast": label,
                "OR": or_, "OR_lo": lo, "OR_hi": hi, "p_raw": p,
                "rd": rd, "rd_lo": rd_lo, "rd_hi": rd_hi, "p_perm": p_perm,
                "gee_converged": conv_d if outcome == "dd_a" else conv_a,
            })
        p_holm = _holm([r["p_raw"] for r in fam])
        for r, ph in zip(fam, p_holm):
            r["p_holm"] = ph
        rows.extend(fam)
    return pd.DataFrame(rows)


def interaction_table(df):
    """Pooled GEE with condition x model interaction; joint Wald test."""
    rows = []
    if df["model"].nunique() < 2:
        return pd.DataFrame([{"note": "interaction skipped: <2 models"}])
    for outcome in ["chose_distractor", "is_correct"]:
        dd = df.copy()
        dd["y"] = dd[outcome].astype(int)
        dd["condition"] = dd["condition"].astype(str)
        dd["model_f"] = dd["model"].astype(str)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = smf.gee("y ~ C(condition)*C(model_f)", "case_id", data=dd,
                          family=sm.families.Binomial(), cov_struct=Exchangeable()).fit()
            conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        # Joint Wald on all condition x model interaction coefficients (reference-invariant,
        # and the count of terms adapts automatically to more models/cases).
        names = res.model.exog_names
        inter_idx = [i for i, n in enumerate(names)
                     if ":" in n and "condition" in n and "model_f" in n]
        if not inter_idx:
            rows.append({"outcome": outcome, "note": "no interaction terms"})
            continue
        R = np.zeros((len(inter_idx), len(names)))
        for r, i in enumerate(inter_idx):
            R[r, i] = 1.0
        wt = res.wald_test(R, scalar=True)
        rows.append({
            "outcome": outcome, "term": "C(condition):C(model)",
            "wald_stat": float(wt.statistic), "df": int(len(inter_idx)),
            "p_value": float(wt.pvalue), "gee_converged": conv,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inferential analysis.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "raw")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "analysis")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true", help="Write tables without printing their contents.")
    args = parser.parse_args(argv)

    df = load_results(args.results_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df)} calls | models: {sorted(df['model'].unique())} | "
          f"cases: {df['case_id'].nunique()} | bootstrap={args.n_boot} perm={args.n_perm}")

    primary = primary_table(df, args.n_boot, args.n_perm)
    secondary = secondary_table(df, args.n_boot, args.n_perm)
    interaction = interaction_table(df)

    primary.to_csv(args.out / "inference_primary.csv", index=False)
    secondary.to_csv(args.out / "inference_secondary.csv", index=False)
    interaction.to_csv(args.out / "inference_interaction.csv", index=False)

    pd.set_option("display.width", 160, "display.max_columns", 30)

    if args.quiet:
        print(f"Wrote GEE and permutation tests to {args.out}")
        return

    def pct(d, cols):
        d = d.copy()
        for c in cols:
            if c in d:
                d[c] = (d[c] * 100).round(1)
        return d

    print("\n=== PRIMARY: severe-neutral DD_a lift, per model (two-sided a=0.05, no cross-model correction) ===")
    p = primary.copy()
    p["OR"] = p["OR"].round(2); p["OR_lo"] = p["OR_lo"].round(2); p["OR_hi"] = p["OR_hi"].round(2)
    p = pct(p, ["rd_lift", "rd_lo", "rd_hi"])
    for c in ["p_gee", "p_perm", "p_wilcoxon"]:
        p[c] = p[c].map(lambda v: f"{v:.2g}" if pd.notna(v) else "na")
    print(p[["model", "OR", "OR_lo", "OR_hi", "p_gee", "rd_lift", "rd_lo", "rd_hi",
             "rd_excludes_0", "p_perm", "p_wilcoxon", "wilcoxon_n_nonzero", "gee_converged"]].to_string(index=False))

    print("\n=== SECONDARY (Holm within each model) ===")
    s = secondary.copy()
    s["OR"] = s["OR"].round(2)
    s = pct(s, ["rd", "rd_lo", "rd_hi"])
    for c in ["p_raw", "p_holm", "p_perm"]:
        s[c] = s[c].map(lambda v: f"{v:.2g}" if pd.notna(v) else "na")
    print(s[["model", "outcome", "contrast", "OR", "rd", "rd_lo", "rd_hi",
             "p_raw", "p_holm", "p_perm"]].to_string(index=False))

    print("\n=== POOLED condition x model interaction (do models differ?) ===")
    print(interaction.to_string(index=False))

    print(f"\nWrote inference_*.csv to {args.out}")


if __name__ == "__main__":
    main()
