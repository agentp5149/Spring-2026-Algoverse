"""
Figure 1: In-distribution coverage comparison across all three domains.
Conformal vs baselines at 90% and 95% target levels.

FIX: mol_ensemble was hardcoded as 0.0 but deep ensemble was never run
on the molecular surrogate. Set to None so it is omitted from the figure,
consistent with how weather MC/Bayesian and mol Bayesian are handled.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np
import os

_liberation_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
for _fname in ["LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
               "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"]:
    _fpath = os.path.join(_liberation_dir, _fname)
    if os.path.exists(_fpath):
        fm.fontManager.addfont(_fpath)

plt.rcParams.update({
    "font.family":     "Liberation Serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

C_CONFORMAL  = "#495362"
C_ENSEMBLE   = "#4b6d60"
C_MCDROPOUT  = "#8997a2"
C_BAYESIAN   = "#cfe0e3"
C_NOMINAL    = "#d1dfe6"

mol_conformal = np.mean([0.9530, 0.9525, 0.9524, 0.9501]) * 100
mol_mc        = np.mean([0.5561, 0.9631, 0.9978, 0.9823]) * 100
mol_ensemble  = None   # deep ensemble was never run on molecular surrogate
mol_bayesian  = None

weather_conformal_90 = 92.3
weather_conformal_95 = 95.8
weather_ensemble     = 0.0
weather_mc           = None
weather_bayesian     = None

pk_conformal  = 93.3
pk_ensemble   = 0.0
pk_mc         = 0.0
pk_bayesian   = 0.0

panels = {
    "90% target": {
        "nominal": 90,
        "Conformal\n(ours)":  [mol_conformal,       weather_conformal_90, pk_conformal],
        "Deep\nEnsemble":     [mol_ensemble,         weather_ensemble,     pk_ensemble],
        "MC\nDropout":        [mol_mc,               None,                 pk_mc],
        "Bayesian\nVI":       [None,                 None,                 pk_bayesian],
    },
    "95% target": {
        "nominal": 95,
        "Conformal\n(ours)":  [mol_conformal,       weather_conformal_95, pk_conformal],
        "Deep\nEnsemble":     [mol_ensemble,         weather_ensemble,     pk_ensemble],
        "MC\nDropout":        [mol_mc,               None,                 pk_mc],
        "Bayesian\nVI":       [None,                 None,                 pk_bayesian],
    },
}

colors = {
    "Conformal\n(ours)": C_CONFORMAL,
    "Deep\nEnsemble":    C_ENSEMBLE,
    "MC\nDropout":       C_MCDROPOUT,
    "Bayesian\nVI":      C_BAYESIAN,
}

domains   = ["Molecular\n(MD17 avg)", "Weather\n(ERA5)", "Pharmacokinetics\n(Neural ODE)"]
MIN_HEIGHT = 1.5

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig.patch.set_facecolor("white")

for ax, (panel_title, panel_data) in zip(axes, panels.items()):
    nominal  = panel_data["nominal"]
    methods  = [k for k in panel_data if k != "nominal"]
    n_m      = len(methods)
    x        = np.arange(3)
    bw       = 0.16
    offsets  = np.linspace(-(n_m - 1) / 2 * bw, (n_m - 1) / 2 * bw, n_m)

    for offset, method in zip(offsets, methods):
        vals  = panel_data[method]
        color = colors[method]
        for di, val in enumerate(vals):
            if val is None:
                continue
            is_zero = (val == 0.0)
            bar_h   = MIN_HEIGHT if is_zero else val
            ax.bar(x[di] + offset, bar_h, width=bw,
                   color=color, zorder=3,
                   linewidth=0.5 if is_zero else 0,
                   edgecolor="white")
            if is_zero:
                ax.text(x[di] + offset, 1.8, "0%",
                        ha="center", va="bottom",
                        fontsize=8, color=color)

    ax.axhline(nominal, color=C_NOMINAL, linewidth=1.5,
               linestyle="--", zorder=2)
    ax.set_title(panel_title, fontweight="normal", pad=10, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 90, 95, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Empirical coverage (%)")

legend_handles = [
    mpatches.Patch(color=C_CONFORMAL, label="Conformal (ours)"),
    mpatches.Patch(color=C_ENSEMBLE,  label="Deep Ensemble"),
    mpatches.Patch(color=C_MCDROPOUT, label="MC Dropout"),
    mpatches.Patch(color=C_BAYESIAN,  label="Bayesian VI"),
    plt.Line2D([0], [0], color=C_NOMINAL, linewidth=1.5,
               linestyle="--", label="Nominal target"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "In-distribution coverage: conformal prediction vs baselines",
    fontsize=14, fontweight="normal", y=1.01, color="#222",
)
fig.text(0.5, -0.11,
    "Coverage measured under strict trajectory metric (all time steps covered simultaneously). "
    "† MC Dropout per-snapshot coverage ranges from 55.6% to 99.8% across MD17 molecules. "
    "Deep Ensemble was not run on the molecular surrogate (not applicable).",
    ha="center", fontsize=9, color="#888")

plt.tight_layout()
plt.savefig("figure images/figure1_coverage_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved figure images/figure1_coverage_comparison.png")
