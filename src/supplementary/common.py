"""Case-level counts and resampling for the supplementary analyses."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import CANONICAL_CONDITIONS as CONDITIONS
from ..analysis import PROJECT_ROOT, SEED, TRIGGERS, _bootstrap_ci, _paired_sign_flip_p, load_results

RAW_DIR = PROJECT_ROOT / "results" / "raw"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
OUT_DIR = PROJECT_ROOT / "results" / "supplementary"
METADATA_DIR = RAW_DIR / "metadata"
FIG_DIR = PROJECT_ROOT / "figures"
N_BOOT = 2000
N_PERM = 10000

MODEL_ORDER = ["gemini-3.1-pro-preview", "gpt-5.4", "qwen3.6-plus", "deepseek-v4-pro", "glm-5.1"]
MODEL_NAME = {
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro-Preview",
    "glm-5.1": "GLM-5.1",
    "gpt-5.4": "GPT-5.4",
    "qwen3.6-plus": "Qwen-3.6-Plus",
}


def load_calls(extra: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = load_results(RAW_DIR, extra_fields=extra)
    frame["other_wrong"] = frame["parseable"] & ~frame["is_correct"] & ~frame["chose_distractor"]
    return frame


def ordered_models(df: pd.DataFrame) -> list:
    present = list(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present] + sorted(m for m in present if m not in MODEL_ORDER)


def case_counts(d: pd.DataFrame, outcomes=("is_correct", "chose_distractor"), conditions=CONDITIONS):
    """Per-case counts for the calls in `d` (one model).

    Returns (cases, counts): `cases` is the sorted array of case IDs and
    counts[condition][name] is an array aligned with it. counts[condition]["n"]
    is the number of parseable calls (the denominator of every rate).
    """
    cases = np.array(sorted(d["case_id"].unique()))
    pos = {c: i for i, c in enumerate(cases)}
    counts = {}
    for cond in conditions:
        sub = d[d["condition"] == cond]
        idx = np.array([pos[c] for c in sub["case_id"]], dtype=int)
        entry = {"n": np.bincount(idx, weights=sub["parseable"].to_numpy(dtype=float), minlength=len(cases))}
        for name in outcomes:
            entry[name] = np.bincount(idx, weights=sub[name].to_numpy(dtype=float), minlength=len(cases))
        counts[cond] = entry
    return cases, counts


def subset_counts(counts: dict, mask: np.ndarray) -> dict:
    """Restrict per-case count arrays to the cases selected by a boolean mask."""
    return {cond: {k: v[mask] for k, v in entry.items()} for cond, entry in counts.items()}


SELECT_REPS = (1, 2, 3, 4, 5)       # neutral repetitions that define the subset
BASELINE_REPS = (6, 7, 8, 9, 10)    # held-out neutral repetitions that give the baseline


def split_half_calls(dm: pd.DataFrame, select_reps=SELECT_REPS, baseline_reps=BASELINE_REPS) -> pd.DataFrame:
    """Calls of the high-consistency subset of one model: held-out neutral draws plus all mild and severe draws.

    A case belongs to the subset if every neutral repetition in `select_reps` is correct. Only the neutral
    repetitions in `baseline_reps` are kept as the neutral condition of the subset.
    """
    neutral = dm[dm["condition"] == "neutral"]
    selected = neutral[neutral["repetition"].isin(select_reps)]
    groups = selected.groupby("case_id")
    all_correct = groups["is_correct"].all()
    complete = groups["repetition"].nunique().eq(len(select_reps))
    keep = all_correct.index[all_correct & complete]
    in_subset = dm["case_id"].isin(keep)
    held_out = (dm["condition"] == "neutral") & dm["repetition"].isin(baseline_reps)
    pressure = dm["condition"].isin(TRIGGERS)
    return dm[in_subset & (held_out | pressure)]


def rate(counts: dict, cond: str, outcome: str, idx=None) -> float:
    """Pooled rate of `outcome` in `cond` (sum of events / sum of parseable calls)."""
    num, den = counts[cond][outcome], counts[cond]["n"]
    if idx is not None:
        num, den = num[idx], den[idx]
    total = den.sum()
    return float(num.sum() / total) if total > 0 else float("nan")


def boot_ci(stat, n_cases: int, seed: int, n_boot: int = N_BOOT):
    return _bootstrap_ci(n_cases, stat, n_boot, seed)


def rate_ci(counts: dict, cond: str, outcome: str, seed: int):
    """Rate with cluster-bootstrap CI: (estimate, lo, hi)."""
    n = len(counts[cond]["n"])
    lo, hi = boot_ci(lambda s: rate(counts, cond, outcome, s), n, seed)
    return rate(counts, cond, outcome), lo, hi


def rate_diff(counts: dict, outcome: str, cond_a: str, cond_b: str, seed: int):
    """Risk difference rate(cond_b) - rate(cond_a) with cluster-bootstrap CI: (estimate, lo, hi)."""
    n = len(counts[cond_a]["n"])
    stat = lambda s: rate(counts, cond_b, outcome, s) - rate(counts, cond_a, outcome, s)
    lo, hi = boot_ci(stat, n, seed)
    return stat(np.arange(n)), lo, hi


def per_case_rate(counts: dict, cond: str, outcome: str) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts[cond]["n"] > 0, counts[cond][outcome] / counts[cond]["n"], np.nan)


def perm_p(counts: dict, outcome: str, cond_a: str, cond_b: str, seed: int, n_perm: int = N_PERM) -> float:
    """Two-sided paired sign-flip permutation p value for the mean per-case difference (cond_b - cond_a)."""
    a, b = counts[cond_a], counts[cond_b]
    return _paired_sign_flip_p(a[outcome], a["n"], b[outcome], b["n"], n_perm, seed)


def pct(x, digits: int = 2):
    """Proportion -> percent / percentage points, rounded."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100.0 * float(x), digits)


def save_table(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(path.relative_to(PROJECT_ROOT))
    return path
