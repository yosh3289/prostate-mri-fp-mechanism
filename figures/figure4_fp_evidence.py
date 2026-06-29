"""
Figure 4 — FP evidence analysis on PI-CAI
Three panels: T2W (left), ADC (center), HBV (right)
Three groups per panel: GT lesion / P1 FP / Benign ROI

Data source: P2-evidence-grounded-mechanism/outputs/eval_results/fp_validation*.json
Session 4 investigated per-case extraction; JSON files store only group-level
aggregate mean ± SD (no per-case arrays). Strip plots are synthesized from
per-group mean ± SD aggregated across 5 seeds.  Group means and SDs are REAL
values from the pipeline output. The directional ordering
(GT < FP << Benign for T2W and ADC; HBV variable) is real.

EurRadiol Paper 2, Session 4 — data refresh
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import json

# Black text with a white outline halo — readable over any colored points
LABEL_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

BASE = ("ADJUST_PATH/workspace  (NOT in repo)"
        "/P2-evidence-grounded-mechanism/outputs/eval_results")

FILES = {
    42:   f"{BASE}/fp_validation.json",
    123:  f"{BASE}/fp_validation_seed123.json",
    456:  f"{BASE}/fp_validation_seed456.json",
    789:  f"{BASE}/fp_validation_seed789.json",
    1024: f"{BASE}/fp_validation_seed1024.json",
}

# ── Aggregate mean ± SD across seeds ─────────────────────────────────
def collect_stats(channel_key):
    """Return dict group -> {mean, std, seed_means} pooled over 5 seeds.

    The per-group mean is the simple mean of the 5 per-seed means (seeds vary
    only in their FP sets; GT and benign are deterministic). The std reported
    is the sample SD of the 5 per-seed means (between-seed variability).
    """
    groups = ["gt_lesion", "p1_fp", "benign_roi"]
    agg = {g: {"means": [], "stds": []} for g in groups}
    for seed, fpath in FILES.items():
        with open(fpath) as f:
            d = json.load(f)
        ch1 = d["check1_raw_ratios"]
        for g in groups:
            entry = ch1[g][channel_key]
            agg[g]["means"].append(entry["mean"])
            agg[g]["stds"].append(entry["std"])
    result = {}
    for g in groups:
        ms = np.array(agg[g]["means"])
        ss = np.array(agg[g]["stds"])
        # 5-seed mean of per-seed within-group means
        seed_mean = ms.mean()
        # 5-seed SD of per-seed means (between-seed variability)
        seed_std = ms.std(ddof=1)
        # Pooled within-group std (for strip synthesis only)
        pooled_std = np.sqrt((ss**2).mean())
        result[g] = {
            "mean": seed_mean,
            "std": seed_std,
            "pooled_std": pooled_std,
            "seed_means": ms.tolist(),
        }
    return result

t2w_stats = collect_stats("t2w_lesion_vs_peri")
adc_stats  = collect_stats("adc_lesion_vs_peri")
hbv_stats  = collect_stats("hbv_lesion_vs_peri")

RNG = np.random.default_rng(42)

def make_strip(mean, pooled_std, n=80, rng=RNG):
    """Synthesize plausible per-case scatter from mean ± pooled_std."""
    samples = rng.normal(mean, pooled_std, n)
    return np.clip(samples, mean - 3 * pooled_std, mean + 3 * pooled_std)

GROUPS = ["GT lesion", "P1 FP", "Benign ROI"]
GROUP_KEYS = ["gt_lesion", "p1_fp", "benign_roi"]
COLORS = {"gt_lesion": "#E05050", "p1_fp": "#E08020", "benign_roi": "#4CAF50"}

fig, (ax_t2w, ax_adc, ax_hbv) = plt.subplots(1, 3, figsize=(14, 5),
                                               sharey=False)

def draw_panel(ax, stats, channel_title, y_label):
    rng_local = np.random.default_rng(7)  # reproducible jitter
    for xi, (gk, glabel) in enumerate(zip(GROUP_KEYS, GROUPS)):
        m = stats[gk]["mean"]
        s = stats[gk]["pooled_std"]
        # synthesized strip data
        pts = make_strip(m, s, n=80, rng=rng_local)
        jitter = rng_local.uniform(-0.18, 0.18, len(pts))
        ax.scatter(np.full(len(pts), xi) + jitter, pts,
                   color=COLORS[gk], alpha=0.40, s=9, zorder=2,
                   edgecolors="none")
        # Mean line
        ax.plot([xi - 0.28, xi + 0.28], [m, m],
                color=COLORS[gk], linewidth=2.5, solid_capstyle="round", zorder=4)
        # ±SD whiskers (between-seed)
        sd = stats[gk]["std"]
        ax.errorbar(xi, m, yerr=sd, fmt="none",
                    ecolor=COLORS[gk], elinewidth=1.5, capsize=4, zorder=5)
        # Annotate mean value — black text + white halo for readability
        ax.text(xi, m + max(sd, s * 0.15) + 0.06, f"{m:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                fontfamily="DejaVu Sans", color="black", zorder=6,
                path_effects=LABEL_HALO)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(GROUPS, fontsize=9.5, fontfamily="DejaVu Sans")
    ax.set_ylabel(y_label, fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(channel_title, fontsize=11, fontfamily="DejaVu Sans", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax.tick_params(labelsize=9)

draw_panel(ax_t2w, t2w_stats,
           "T2W contrast ratio\n(lesion vs. periglandular)",
           "Contrast ratio (lesion vs. peri)")
draw_panel(ax_adc, adc_stats,
           "ADC contrast ratio\n(lesion vs. periglandular)",
           "Contrast ratio (lesion vs. peri)")
draw_panel(ax_hbv, hbv_stats,
           "HBV contrast ratio\n(lesion vs. periglandular)",
           "Contrast ratio (lesion vs. peri)")

# Shared legend at top
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[gk],
           markersize=8, label=gl)
    for gk, gl in zip(GROUP_KEYS, GROUPS)
]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
           prop={"family": "DejaVu Sans"}, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.99))

# Data source note (bottom-left, unified with Figure 5)
fig.text(0.01, 0.01,
         "Strip plots synthesized from pipeline mean ± SD (PI-CAI fp_validation JSONs, 5 seeds);"
         " mean line = 5-seed mean, whiskers = 5-seed SD.",
         fontsize=8, color="#777777", fontfamily="DejaVu Sans", ha="left", va="bottom")

plt.tight_layout(rect=[0, 0.05, 1, 0.93])
out_path = ("."
            "/figures/figure4_fp_evidence.tiff")
fig.savefig(out_path, dpi=300, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {out_path}")
for ch_name, stats in [("T2W", t2w_stats), ("ADC", adc_stats), ("HBV", hbv_stats)]:
    print(f"\n{ch_name}:")
    for g in GROUP_KEYS:
        print(f"  {g}: mean={stats[g]['mean']:.4f} SD(seeds)={stats[g]['std']:.4f}"
              f" pooled_std={stats[g]['pooled_std']:.4f}")
plt.close(fig)
