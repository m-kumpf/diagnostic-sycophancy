"""Publication figures for distractor selection and diagnostic accuracy."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .supplementary.common import (ANALYSIS_DIR, CONDITIONS, FIG_DIR, MODEL_NAME, load_calls,
                                   ordered_models, save_table, split_half_calls)

matplotlib.rcParams["pdf.fonttype"] = 42
PASTELS = ["#8FCBA6", "#8FB8DE", "#B79FD4", "#E8B7A0", "#9FD4CE", "#D4A6C8"]


def lighten(hex_color: str, frac: float) -> tuple:
    """Blend a hex color toward white by `frac` in [0,1] (0 = original)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * frac)


def stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _model_colors(models):
    palette_order = sorted(set(models))
    return {m: PASTELS[i % len(PASTELS)] for i, m in enumerate(palette_order)}


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="both", color="#ECECEC", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _spread_labels(ys, min_gap):
    """Displace end-of-line labels so that neighbours are at least `min_gap` apart (order preserved)."""
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    new = list(ys)
    prev = None
    for i in order:
        if prev is None:
            prev = new[i]
            continue
        if new[i] > prev - min_gap:
            new[i] = prev - min_gap
        prev = new[i]
    return new


def condition_rates(d: pd.DataFrame, outcome: str) -> np.ndarray:
    """Pooled rate (%) of `outcome` per condition, over parseable calls."""
    out = []
    for cond in CONDITIONS:
        sub = d[(d["condition"] == cond) & d["parseable"]]
        out.append(100.0 * sub[outcome].mean() if len(sub) else np.nan)
    return np.array(out)


def draw_lines(ax, values, models, colours, min_gap, leader_tol):
    """One line per model across the three conditions, with spread end-of-line labels."""
    x = list(range(len(CONDITIONS)))
    last = []
    for m in models:
        ax.plot(x, values[m], "-o", color=colours[m], lw=2.5, markersize=8,
                markeredgecolor="white", label=MODEL_NAME.get(m, m), zorder=3)
        last.append(float(values[m][-1]))
    for m, real_y, lab_y in zip(models, last, _spread_labels(last, min_gap)):
        ax.annotate(MODEL_NAME.get(m, m), (len(CONDITIONS) - 1, lab_y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=colours[m])
        if abs(lab_y - real_y) > leader_tol:
            ax.plot([len(CONDITIONS) - 1, len(CONDITIONS) - 1 + 0.10], [real_y, lab_y],
                    color=colours[m], lw=0.8, alpha=0.4, zorder=2)


def figure_2(rates, primary, secondary):
    conds = CONDITIONS
    models = ordered_models(rates)
    mcol = _model_colors(models)
    x = np.arange(len(models))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(2.0 * len(models) + 3, 5.2))
    _style(ax)
    bar_x = {}  # (model, cond) -> x position
    for j, cond in enumerate(conds):
        shade = 0.55 - 0.27 * (j / max(len(conds) - 1, 1))
        for i, m in enumerate(models):
            r = rates[(rates.model == m) & (rates.condition == cond)]
            if r.empty:
                continue
            xb = x[i] + j * w
            bar_x[(m, cond)] = xb
            v = r.dd_a_rate.iloc[0] * 100
            err = [[v - r.dd_a_lo.iloc[0] * 100], [r.dd_a_hi.iloc[0] * 100 - v]]
            ax.bar(xb, v, w, color=lighten(mcol[m], shade), edgecolor="white",
                   yerr=err, capsize=3, zorder=3,
                   label=cond if i == 0 else None)

    primary_p = primary.set_index("model").p_gee
    secondary_p = secondary[secondary.outcome == "dd_a"].set_index(["model", "contrast"]).p_holm
    overall_top = rates.dd_a_hi.max() * 100

    def bracket(x0, x1, y, text):
        tick = overall_top * 0.03 + 0.4
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color="#666666", lw=1.1, zorder=4)
        ax.text((x0 + x1) / 2, y + 0.3, text, ha="center", va="bottom", fontsize=10, color="#333333")

    for model in models:
        upper = rates[rates.model == model].dd_a_hi.max() * 100
        y1 = upper + overall_top * 0.06 + 1.5
        y2 = upper + overall_top * 0.20 + 4.5
        neutral, mild, severe = (bar_x[(model, condition)] for condition in CONDITIONS)
        bracket(neutral, mild, y1, stars(secondary_p.loc[(model, "mild-neutral")]))
        bracket(mild, severe, y1, stars(secondary_p.loc[(model, "severe-mild")]))
        bracket(neutral, severe, y2, stars(primary_p.loc[model]))

    ax.set_xticks(x + w * (len(conds) - 1) / 2)
    ax.set_xticklabels([MODEL_NAME[m] for m in models], rotation=12, ha="right")
    ax.set_ylim(0, overall_top + 16)
    ax.set_ylabel("agreement with distractor (%)")
    ax.set_title("Distractor Agreement")
    grey_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#D5D5D5", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#999999", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#555555", edgecolor="white"),
    ]
    leg = ax.legend(grey_handles, conds,
                    title="condition", frameon=False, loc="upper center")
    ax.add_artist(leg)
    ax.text(0.99, 0.97, "ns p≥.05   * p<.05   ** p<.01   *** p<.001",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_2.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_3():
    df = load_calls()
    models = ordered_models(df)
    colours = _model_colors(models)

    dda_all, acc_all, dda_sub, acc_sub, rows = {}, {}, {}, {}, []
    for m in models:
        dm = df[df["model"] == m]
        sub = split_half_calls(dm)
        dda_all[m], acc_all[m] = condition_rates(dm, "chose_distractor"), condition_rates(dm, "is_correct")
        dda_sub[m], acc_sub[m] = condition_rates(sub, "chose_distractor"), condition_rates(sub, "is_correct")
        for panel, n, dda, acc in (("all cases", dm["case_id"].nunique(), dda_all[m], acc_all[m]),
                                   ("high-consistency cases", sub["case_id"].nunique(), dda_sub[m], acc_sub[m])):
            for j, cond in enumerate(CONDITIONS):
                rows.append({"model": MODEL_NAME.get(m, m), "panel": panel, "n_cases": n, "condition": cond,
                             "distractor_selection_pct": round(float(dda[j]), 2), "accuracy_pct": round(float(acc[j]), 2)})

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[9, 9, 1], hspace=0.10, wspace=0.05)
    axL_dda = fig.add_subplot(gs[0, 0])
    axR_dda = fig.add_subplot(gs[0, 1], sharex=axL_dda, sharey=axL_dda)
    axL_top = fig.add_subplot(gs[1, 0], sharex=axL_dda)
    axR_top = fig.add_subplot(gs[1, 1], sharex=axL_dda, sharey=axL_top)
    axL_bot = fig.add_subplot(gs[2, 0], sharex=axL_dda)
    axR_bot = fig.add_subplot(gs[2, 1], sharex=axL_dda, sharey=axL_bot)
    for ax in (axL_dda, axR_dda, axL_top, axR_top, axL_bot, axR_bot):
        _style(ax)

    # Top row: suggested-distractor selection.
    draw_lines(axL_dda, dda_all, models, colours, min_gap=1.6, leader_tol=0.05)
    draw_lines(axR_dda, dda_sub, models, colours, min_gap=1.6, leader_tol=0.05)
    for ax in (axL_dda, axR_dda):
        ax.set_ylim(-1, 30)
        ax.tick_params(labelbottom=False)
    axL_dda.set_ylabel("Distractor selection rate (%)")
    axR_dda.tick_params(labelleft=False)
    axL_dda.set_title("All cases")
    axR_dda.set_title("High-consistency cases (held-out neutral baseline)")

    # Bottom rows: accuracy with a broken y-axis.
    draw_lines(axL_top, acc_all, models, colours, min_gap=2.5, leader_tol=0.2)
    draw_lines(axR_top, acc_sub, models, colours, min_gap=2.5, leader_tol=0.2)
    axL_top.set_ylim(55, 102)
    axL_bot.set_ylim(-1, 10)
    n = len(CONDITIONS)
    for ax in (axL_top, axR_top):
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.plot([-0.5, n - 0.3 + 1.1], [55, 55], color="#333", linestyle=":", lw=1.8, zorder=5, clip_on=False)
    for ax in (axL_bot, axR_bot):
        ax.spines["top"].set_visible(False)
        ax.set_yticks([0])
        ax.plot([-0.5, n - 0.3 + 1.1], [10, 10], color="#333", linestyle=":", lw=1.8, zorder=5, clip_on=False)
        ax.set_xticks(list(range(n)))
        ax.set_xticklabels(CONDITIONS)
    axL_dda.set_xlim(-0.3, n - 0.3 + 0.9)
    axR_top.tick_params(labelleft=False)
    axR_bot.tick_params(labelleft=False)
    axL_top.set_ylabel("Accuracy (%)")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_3.pdf", bbox_inches="tight", pad_inches=0.15, metadata={"CreationDate": None})
    plt.close(fig)
    save_table(pd.DataFrame(rows), "figure_3_rates.csv")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rates = pd.read_csv(ANALYSIS_DIR / "descriptive_rates.csv", float_precision="round_trip")
    primary = pd.read_csv(ANALYSIS_DIR / "inference_primary.csv", float_precision="round_trip")
    secondary = pd.read_csv(ANALYSIS_DIR / "inference_secondary.csv", float_precision="round_trip")
    figure_2(rates, primary, secondary)
    figure_3()


if __name__ == "__main__":
    main()
