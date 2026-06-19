"""
Figure 2: Physics projection coverage loss table for MD17 molecules.
Exact projection vs epsilon-relaxed projection at 95% target.

FIX: loss_relaxed used to be a hardcoded dict, never actually read from
molecular_week5.py's output. Both loss_exact and loss_relaxed are now
loaded from real saved results, with explicit alpha checks so this fails
loudly instead of silently plotting numbers from the wrong run.
"""

import torch
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
    else:
        print(f"  WARNING: expected font not found at {_fpath}")

plt.rcParams.update({
    "font.family":     "Liberation Serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

C_FOREST    = "#4b6d60"
C_MID_SLATE = "#8997a2"

BASE = "results/molecular"
molecules  = ["aspirin", "ethanol", "uracil", "malonaldehyde"]
mol_labels = ["Aspirin", "Ethanol", "Uracil", "Malonaldehyde"]

TARGET_ALPHA = 0.05  # 95% target, matching this figure's title

rows = []
for mol in molecules:
    proj = torch.load(
        os.path.join(BASE, f"{mol}_conformal_projected_indist.pt"),
        map_location="cpu", weights_only=False,
    )
    if abs(proj["alpha"] - TARGET_ALPHA) > 1e-6:
        raise ValueError(
            f"{mol}: canonical projected_indist.pt has alpha={proj['alpha']}, "
            f"expected {TARGET_ALPHA} for the 95% target this figure reports. "
            f"Check whether Week 3 was last run with a different alpha order, "
            f"or load the alpha-suffixed file instead."
        )
    loss_exact = proj["coverage_loss_empirical"] * 100

    week5_path = os.path.join(BASE, f"{mol}_week5_ablation.pt")
    if not os.path.exists(week5_path):
        raise FileNotFoundError(
            f"Missing {week5_path}. Run: python molecular_week5.py --all "
            f"before regenerating this figure."
        )
    week5_data = torch.load(week5_path, map_location="cpu", weights_only=False)
    match = [r for r in week5_data["ablation_results"] if abs(r["alpha"] - TARGET_ALPHA) < 1e-6]
    if not match:
        raise ValueError(
            f"{mol}: no alpha={TARGET_ALPHA} entry found in {week5_path}. "
            f"Rerun molecular_week5.py with --alphas including {TARGET_ALPHA}."
        )
    loss_relaxed = match[0]["coverage_loss_relaxed"] * 100

    reduction = (1 - loss_relaxed / loss_exact) * 100 if loss_exact > 0 else 0.0
    rows.append({
        "loss_exact":   loss_exact,
        "loss_relaxed": loss_relaxed,
        "reduction":    reduction,
    })
    print(f"  {mol}: exact={loss_exact:.2f}pp  relaxed={loss_relaxed:.2f}pp  "
          f"reduction={reduction:.0f}%")

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")

x        = np.arange(len(molecules))
bw       = 0.35

exact_vals   = [r["loss_exact"]   for r in rows]
relaxed_vals = [r["loss_relaxed"] for r in rows]

bars_e = ax.bar(x - bw / 2, exact_vals,   bw, color=C_FOREST,    zorder=3)
bars_r = ax.bar(x + bw / 2, relaxed_vals, bw, color=C_MID_SLATE, zorder=3)

for bar, val in zip(bars_e, exact_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"{val:.1f}pp", ha="center", va="bottom",
            fontsize=9.5, color=C_FOREST)

for bar, val in zip(bars_r, relaxed_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"{val:.1f}pp", ha="center", va="bottom",
            fontsize=9.5, color=C_MID_SLATE)

for i, r in enumerate(rows):
    ymax = max(r["loss_exact"], r["loss_relaxed"])
    ax.annotate(f"\u2212{r['reduction']:.0f}%",
                xy=(x[i], ymax + 1.2),
                ha="center", va="bottom",
                fontsize=9, color="#555")

ax.set_xticks(x)
ax.set_xticklabels(mol_labels)
ax.set_ylabel("Coverage loss after projection (pp)")
ax.set_ylim(0, max(exact_vals) + 4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#ccc")
ax.spines["bottom"].set_color("#ccc")
ax.yaxis.grid(True, color="#eee", zorder=0)
ax.set_axisbelow(True)
ax.set_title(
    "Physics projection coverage loss: exact vs \u03b5-relaxed (MD17, 95% target)",
    fontweight="normal", pad=12, color="#222",
)

legend_handles = [
    mpatches.Patch(color=C_FOREST,    label="Exact projection"),
    mpatches.Patch(color=C_MID_SLATE, label="\u03b5-relaxed projection"),
]
ax.legend(handles=legend_handles, frameon=False, loc="upper right")

fig.text(0.5, -0.03,
         "Percentages above bars show relative coverage loss reduction from \u03b5-relaxed vs exact projection.",
         ha="center", fontsize=9, color="#888")

plt.tight_layout()
plt.savefig("figure2_projection_coverage_loss.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved figure2_projection_coverage_loss.png")
