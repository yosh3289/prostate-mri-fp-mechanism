"""
Figure 5 — External validation cross-center replication
Two panels: PI-CAI (left), Prostate158 (right)
Each panel: ADC contrast ratio distributions for GT lesion / FP / Benign
Key message: directional ordering preserved in both cohorts (5/5 seeds)

PI-CAI data: synthesized strip plots from per-group mean±SD
             (fp_validation*.json have aggregated stats, no per-case arrays)
Prostate158 data: real per-case arrays from fp_mimic_seed*.json
EurRadiol Paper 2, Session 3
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import json, os

# Black text with a white outline halo — readable over any colored points
LABEL_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

# ── PI-CAI data (mean±std aggregated over 5 seeds) ───────────────────
PICAI_BASE = ("ADJUST_PATH/workspace  (NOT in repo)"
              "/P2-evidence-grounded-mechanism/outputs/eval_results")
PICAI_FILES = {
    42:   f"{PICAI_BASE}/fp_validation.json",
    123:  f"{PICAI_BASE}/fp_validation_seed123.json",
    456:  f"{PICAI_BASE}/fp_validation_seed456.json",
    789:  f"{PICAI_BASE}/fp_validation_seed789.json",
    1024: f"{PICAI_BASE}/fp_validation_seed1024.json",
}

def picai_adc_stats():
    """Return group -> (mean, std) weighted across seeds."""
    groups = ["gt_lesion", "p1_fp", "benign_roi"]
    agg = {g: {"means": [], "stds": [], "ns": []} for g in groups}
    for seed, fpath in PICAI_FILES.items():
        with open(fpath) as f:
            d = json.load(f)
        ch1 = d["check1_raw_ratios"]
        for g in groups:
            entry = ch1[g]["adc_lesion_vs_peri"]
            agg[g]["means"].append(entry["mean"])
            agg[g]["stds"].append(entry["std"])
            agg[g]["ns"].append(ch1[g]["n"])
    result = {}
    for g in groups:
        ns = np.array(agg[g]["ns"])
        ms = np.array(agg[g]["means"])
        ss = np.array(agg[g]["stds"])
        w = ns / ns.sum()
        wmean = np.dot(w, ms)
        wstd  = np.sqrt(np.dot(w, ss**2 + (ms - wmean)**2))
        result[g] = {"mean": wmean, "std": wstd}
    return result

# ── Prostate158 data (real per-case arrays from all 5 seeds) ─────────
P158_BASE = "./experiments/prostate158_p2b"  # adjust to repo experiments/ dir
P158_SEEDS = [42, 123, 456, 789, 1024]

def p158_adc_arrays():
    """Return group -> np.array of per-case ADC contrast ratios (all seeds pooled).
    NaN values (cases without FP/benign ROIs) are filtered out."""
    gt_vals, fp_vals, benign_vals = [], [], []
    for S in P158_SEEDS:
        fpath = f"{P158_BASE}/fp_mimic_seed{S}.json"
        with open(fpath) as f:
            d = json.load(f)
        for case in d["per_case"]:
            v_gt = case.get("adc_gt")
            v_fp = case.get("adc_fp")
            v_bn = case.get("adc_benign")
            if v_gt is not None and not np.isnan(v_gt):
                gt_vals.append(v_gt)
            if v_fp is not None and not np.isnan(v_fp):
                fp_vals.append(v_fp)
            if v_bn is not None and not np.isnan(v_bn):
                benign_vals.append(v_bn)
    return {"gt_lesion": np.array(gt_vals),
            "p1_fp":     np.array(fp_vals),
            "benign_roi": np.array(benign_vals)}

# ── Plotting ──────────────────────────────────────────────────────────
GROUPS = ["GT lesion", "P1 FP", "Benign ROI"]
GROUP_KEYS = ["gt_lesion", "p1_fp", "benign_roi"]
COLORS = {"gt_lesion": "#E05050", "p1_fp": "#E08020", "benign_roi": "#4CAF50"}
CLIP_QUANTILE = 0.02   # clip extreme outliers for better visualization

fig, (ax_picai, ax_p158) = plt.subplots(1, 2, figsize=(10, 5))

RNG = np.random.default_rng(42)

def make_strip_from_stats(mean, std, n=80):
    samples = RNG.normal(mean, std, n)
    return np.clip(samples, mean - 3*std, mean + 3*std)

def draw_panel(ax, data_dict, title, data_note):
    """
    data_dict: group_key -> np.array (real) or dict with mean/std (synthesized)
    """
    for xi, (gk, glabel) in enumerate(zip(GROUP_KEYS, GROUPS)):
        val = data_dict[gk]
        if isinstance(val, np.ndarray):
            pts = val.copy()
            # Clip extreme outliers for visual clarity
            lo, hi = np.quantile(pts, CLIP_QUANTILE), np.quantile(pts, 1 - CLIP_QUANTILE)
            pts = pts[(pts >= lo) & (pts <= hi)]
            median_val = np.median(pts)
        else:  # synthesized from stats
            pts = make_strip_from_stats(val["mean"], val["std"])
            median_val = val["mean"]  # use mean as surrogate

        jitter = RNG.uniform(-0.18, 0.18, len(pts))
        ax.scatter(np.full(len(pts), xi) + jitter, pts,
                   color=COLORS[gk], alpha=0.45, s=10, zorder=2,
                   edgecolors="none")
        ax.plot([xi - 0.28, xi + 0.28], [median_val, median_val],
                color=COLORS[gk], linewidth=2.5, solid_capstyle="round", zorder=4)
        ax.text(xi, median_val + 0.04, f"{median_val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                fontfamily="DejaVu Sans", color="black", zorder=6,
                path_effects=LABEL_HALO)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(GROUPS, fontsize=9.5, fontfamily="DejaVu Sans")
    ax.set_ylabel("ADC contrast ratio (lesion vs. peri)", fontsize=10,
                  fontfamily="DejaVu Sans")
    ax.set_title(title, fontsize=11, fontfamily="DejaVu Sans", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax.tick_params(labelsize=9)
    # Data note (per-panel provenance)
    ax.text(0.02, 0.02, data_note, transform=ax.transAxes,
            fontsize=8, color="#777777", fontfamily="DejaVu Sans", va="bottom")

# PI-CAI panel
picai_stats = picai_adc_stats()
picai_dict = {gk: picai_stats[gk] for gk in GROUP_KEYS}
draw_panel(ax_picai, picai_dict,
           "PI-CAI (primary cohort, n = 300)",
           "Synthesized from mean±SD\n(5-seed aggregate)")

# Prostate158 panel
p158_arrays = p158_adc_arrays()
draw_panel(ax_p158, p158_arrays,
           "Prostate158 (external cohort, n = 158)",
           "Real per-case values\n(5 seeds × 158 cases pooled)")

# Shared legend at top (banner removed; space reclaimed)
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[gk],
           markersize=8, label=gl)
    for gk, gl in zip(GROUP_KEYS, GROUPS)
]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
           prop={"family": "DejaVu Sans"}, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.99))

# Key message relocated to bottom-left (unified with Figure 4 note placement)
fig.text(0.01, 0.008,
         "Directional ordering preserved in both cohorts (5/5 seeds): GT ≈ FP < Benign in ADC contrast.",
         ha="left", va="bottom", fontsize=8, fontfamily="DejaVu Sans", color="#333333")

plt.tight_layout(rect=[0, 0.05, 1, 0.94])
out_path = ("."
            "/figures/figure5_external_val.tiff")
fig.savefig(out_path, dpi=300, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {out_path}")

# Print stats for verification
print("\nPI-CAI ADC stats (5-seed agg):")
for gk, gl in zip(GROUP_KEYS, GROUPS):
    print(f"  {gl}: mean={picai_stats[gk]['mean']:.3f}, std={picai_stats[gk]['std']:.3f}")

print("\nProstate158 ADC arrays (all seeds pooled):")
for gk, gl in zip(GROUP_KEYS, GROUPS):
    arr = p158_arrays[gk]
    print(f"  {gl}: n={len(arr)}, median={np.median(arr):.3f}, mean={np.mean(arr):.3f}")

plt.close(fig)
