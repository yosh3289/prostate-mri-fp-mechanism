"""
Figure 6 — P2a vs. baselines (panel a) and FP-stratification (panel b).
EurRadiol Paper 2, Session 7

Panel (a): Grouped bar chart comparing P2a refinement head vs the simple
           threshold-sweep baseline at matched case-sensitivity (target
           CaseSens ≈ 0.94), on PI-CAI fold-0 validation and Prostate158.
           5-seed mean ± SD; +X.X pp annotation on the P2a-over-baseline gap.

Panel (b): Grouped bar chart of per-tier suppression rates across
           {high, mid, low} cosine-similarity tertiles for both cohorts.
           Visual message: uniform suppression across tiers (FP mechanism
           is not selectively resolving "imaging-mimic" FPs).

Data sources
------------
  experiments/baselines/aggregate_comparison.json
  experiments/fp_stratification/stratification_aggregate.json
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
BASELINES_JSON = ROOT / "experiments/baselines/aggregate_comparison.json"
STRAT_JSON = ROOT / "experiments/fp_stratification/stratification_aggregate.json"

COLOR_P2A = "#1f77b4"
COLOR_BASE = "#888888"
COLOR_PICAI = "#1f77b4"
COLOR_P158  = "#D17317"

# ── Load data ──────────────────────────────────────────────────────────
baselines = json.loads(BASELINES_JSON.read_text())
strat = json.loads(STRAT_JSON.read_text())

cohorts_a = ["PI-CAI", "Prostate158"]
cohort_keys = {"PI-CAI": "picai", "Prostate158": "prostate158"}

# Panel (a) values
base_means = [
    baselines[cohort_keys[c]]["threshold_sweep"]["casespec_matched"]["mean"]
    for c in cohorts_a
]
base_sds = [
    baselines[cohort_keys[c]]["threshold_sweep"]["casespec_matched"]["sd"]
    for c in cohorts_a
]
p2a_means = [
    baselines[cohort_keys[c]]["p2a_reference"]["casespec_matched"]["mean"]
    for c in cohorts_a
]
p2a_sds = [
    baselines[cohort_keys[c]]["p2a_reference"]["casespec_matched"]["sd"]
    for c in cohorts_a
]
deltas = [p - b for p, b in zip(p2a_means, base_means)]

# Panel (b) values — tier order: high, mid, low (user spec order)
tier_order = ["high", "mid", "low"]
supp_picai = [
    strat["picai"]["pooled"]["tiers"][t]["suppression_rate"]
    for t in tier_order
]
supp_p158 = [
    strat["prostate158"]["pooled"]["tiers"][t]["suppression_rate"]
    for t in tier_order
]
picai_overall = strat["picai"]["pooled"]["suppression_rate_overall"]
p158_overall = strat["prostate158"]["pooled"]["suppression_rate_overall"]

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ══════════ Panel (a) ══════════════════════════════════════════════
ax = axes[0]
x = np.arange(len(cohorts_a))
w = 0.38

bars_base = ax.bar(
    x - w/2, base_means, w,
    yerr=base_sds, color=COLOR_BASE, capsize=4,
    edgecolor="white", linewidth=0.8,
    label="Threshold-sweep baseline",
)
bars_p2a = ax.bar(
    x + w/2, p2a_means, w,
    yerr=p2a_sds, color=COLOR_P2A, capsize=4,
    edgecolor="white", linewidth=0.8,
    label="P2a refinement head",
)

# Bar value labels
for bars, means, sds in [(bars_base, base_means, base_sds),
                          (bars_p2a, p2a_means, p2a_sds)]:
    for b, m, s in zip(bars, means, sds):
        ax.text(b.get_x() + b.get_width()/2, m + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=7.5, fontfamily="DejaVu Sans")

# Delta (+X.X pp) annotation brackets
y_top = max(max(p2a_means[i] + p2a_sds[i], base_means[i] + base_sds[i])
            for i in range(len(cohorts_a))) + 0.035
for i, d in enumerate(deltas):
    sign = "+" if d >= 0 else ""
    ax.annotate(
        f"{sign}{d*100:.1f} pp",
        xy=(x[i], y_top),
        ha="center", va="bottom",
        fontsize=9, fontweight="bold", fontfamily="DejaVu Sans",
        color=COLOR_P2A if d >= 0 else "#B03020",
    )
    # Short bracket from baseline top to p2a top on same cohort
    y_bracket = y_top - 0.015
    ax.plot([x[i] - w/2, x[i] + w/2], [y_bracket, y_bracket],
            color="#444444", linewidth=0.9)

ax.set_xticks(x)
ax.set_xticklabels(cohorts_a, fontsize=10, fontfamily="DejaVu Sans")
ax.set_ylabel("Case-level specificity at matched sensitivity (≈ 0.94)",
              fontsize=10, fontfamily="DejaVu Sans")
ax.set_ylim(0, max(y_top + 0.05, 0.18))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
          prop={"family": "DejaVu Sans"})
ax.set_title("(a)  P2a vs. threshold-sweep baseline (5-seed mean ± SD)",
             fontsize=10, fontfamily="DejaVu Sans", loc="left")

# ══════════ Panel (b) ══════════════════════════════════════════════
ax = axes[1]
xb = np.arange(len(tier_order))
wb = 0.38

bars_picai = ax.bar(
    xb - wb/2, supp_picai, wb,
    color=COLOR_PICAI, edgecolor="white", linewidth=0.8,
    label=f"PI-CAI (overall {picai_overall*100:.1f}%)",
)
bars_p158 = ax.bar(
    xb + wb/2, supp_p158, wb,
    color=COLOR_P158, edgecolor="white", linewidth=0.8,
    label=f"Prostate158 (overall {p158_overall*100:.1f}%)",
)

for bars, vals in [(bars_picai, supp_picai), (bars_p158, supp_p158)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v*100:.1f}%", ha="center", va="bottom",
                fontsize=7.5, fontfamily="DejaVu Sans")

# Horizontal reference lines at overall rates
ax.axhline(picai_overall, color=COLOR_PICAI, linestyle=":",
           linewidth=0.9, alpha=0.7)
ax.axhline(p158_overall, color=COLOR_P158, linestyle=":",
           linewidth=0.9, alpha=0.7)

ax.set_xticks(xb)
ax.set_xticklabels([t.capitalize() + "\nsimilarity" for t in tier_order],
                    fontsize=10, fontfamily="DejaVu Sans")
ax.set_ylabel("Per-tier suppression rate",
              fontsize=10, fontfamily="DejaVu Sans")
ax.set_ylim(0, max(max(supp_picai), max(supp_p158)) + 0.08)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

# Annotation: uniform suppression message
ax.text(
    0.02, 0.98,
    "Uniform suppression across tiers\n(rates within ±2 pp of overall)",
    transform=ax.transAxes,
    fontsize=8, va="top", ha="left", fontfamily="DejaVu Sans", fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#BBBBBB", alpha=0.9),
)

ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
          prop={"family": "DejaVu Sans"})
ax.set_title("(b)  Per-tier FP suppression rate by imaging-mimic similarity",
             fontsize=10, fontfamily="DejaVu Sans", loc="left")

plt.tight_layout()

out_path = ROOT / "figures/figure6_baselines_stratification.tiff"
fig.savefig(str(out_path), dpi=300, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {out_path}")
print("Panel (a):")
for i, c in enumerate(cohorts_a):
    print(f"  {c}: base {base_means[i]:.4f}±{base_sds[i]:.4f}  "
          f"P2a {p2a_means[i]:.4f}±{p2a_sds[i]:.4f}  "
          f"Δ = {deltas[i]*100:+.1f} pp")
print("Panel (b):")
for t, p, q in zip(tier_order, supp_picai, supp_p158):
    print(f"  {t}: PI-CAI {p*100:.2f}%  Prostate158 {q*100:.2f}%")
print(f"  PI-CAI overall {picai_overall*100:.2f}%  "
      f"Prostate158 overall {p158_overall*100:.2f}%")
plt.close(fig)
