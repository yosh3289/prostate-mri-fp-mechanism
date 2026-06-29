"""
Figure 3 — P2b explanation feasibility (PI-CAI)
Panel (a): Paired seed-level MAE dot plot (3 variants × 5 seeds)
Panel (b): Per-target R² heatmap (3 variants × 6 targets)
EurRadiol Paper 2, Session 3
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json
import os

BASE = ("ADJUST_PATH/workspace  (NOT in repo)"
        "/P2-evidence-grounded-mechanism/outputs/eval_results")

VARIANTS = ["mask_only", "image_aware", "context_aware"]
VARIANT_LABELS = ["Mask-only", "Image-aware", "Context-aware"]
SEEDS = [42, 123, 456, 789, 1024]

TARGETS = [
    "t2w_lesion_vs_peri",
    "t2w_lesion_vs_contra",
    "adc_lesion_vs_peri",
    "adc_lesion_vs_contra",
    "hbv_lesion_vs_peri",
    "hbv_lesion_vs_contra",
]
TARGET_LABELS = [
    "T2W lesion/peri",
    "T2W lesion/contra",
    "ADC lesion/peri",
    "ADC lesion/contra",
    "HBV lesion/peri",
    "HBV lesion/contra",
]

# ── Load data ──────────────────────────────────────────────────────────
mae_data  = {}  # variant -> list[5 seeds]
f1_data   = {}
r2_data   = {}  # variant -> list[6 targets, mean over 5 seeds]

for v in VARIANTS:
    maes, f1s = [], []
    r2_by_target = {t: [] for t in TARGETS}
    for S in SEEDS:
        path = os.path.join(BASE, f"explanation_{v}_seed{S}.json")
        with open(path) as f:
            d = json.load(f)
        maes.append(d["evidence_metrics"]["overall_mae"])
        f1s.append(d.get("suspicion_macro_f1", np.nan))
        for t in TARGETS:
            r2_by_target[t].append(d["evidence_metrics"]["per_ratio_r2"][t])
    mae_data[v] = np.array(maes)
    f1_data[v]  = np.array(f1s)
    r2_data[v]  = np.array([np.mean(r2_by_target[t]) for t in TARGETS])

# ── Figure ─────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.3))

COLORS = {"mask_only": "#888888", "image_aware": "#E08020", "context_aware": "#1f77b4"}
XPOS   = {"mask_only": 0, "image_aware": 1, "context_aware": 2}
OFFSET_MAP = {0: -0.08, 1: 0.0, 2: 0.08}

# ─── Panel (a): MAE dot plot ───────────────────────────────────────────
# Per-seed dots and the mean±SD summary share the SAME x (= tick) so the
# pairing is unambiguous. The summary is a larger diamond with a dark edge
# drawn on top of the small semi-transparent dots — distinct without moving.
for i, (v, lbl) in enumerate(zip(VARIANTS, VARIANT_LABELS)):
    xs = np.full(len(SEEDS), i) + np.random.default_rng(0).uniform(-0.07, 0.07, len(SEEDS))
    ax_a.scatter(xs, mae_data[v], facecolors=COLORS[v], s=34, zorder=3,
                 alpha=0.45, edgecolors="none")

# Connect seeds across variants with thin grey lines
for seed_idx in range(len(SEEDS)):
    ys = [mae_data[v][seed_idx] for v in VARIANTS]
    ax_a.plot([0, 1, 2], ys, color="#CCCCCC", linewidth=0.7, zorder=1)

# Mean ± SD overlay — same x as dots, larger diamond + dark edge on top
for i, v in enumerate(VARIANTS):
    m, s = np.mean(mae_data[v]), np.std(mae_data[v], ddof=1)
    ax_a.errorbar(i, m, yerr=s, fmt="D", markerfacecolor=COLORS[v],
                  markeredgecolor="#1A1A1A", markeredgewidth=1.2, ecolor="#1A1A1A",
                  markersize=12, capsize=5, linewidth=1.6, zorder=5)
    ax_a.text(i + 0.13, m, f"{m:.3f}", ha="left", va="center",
              fontsize=8.5, fontfamily="DejaVu Sans", color="#1A1A1A", fontweight="bold")

# Δ annotation: context vs mask
m_mask = np.mean(mae_data["mask_only"])
m_ctx  = np.mean(mae_data["context_aware"])
pct    = (m_mask - m_ctx) / m_mask * 100
ax_a.annotate(
    f"−{pct:.1f}% vs mask-only",
    xy=(2, m_ctx), xytext=(2.35, (m_mask + m_ctx) / 2),
    fontsize=8, color=COLORS["context_aware"],
    fontfamily="DejaVu Sans",
    arrowprops=dict(arrowstyle="->,head_width=0.2", color=COLORS["context_aware"],
                    linewidth=1.2),
)

ax_a.set_xticks([0, 1, 2])
ax_a.set_xticklabels(VARIANT_LABELS, fontsize=9, fontfamily="DejaVu Sans")
ax_a.set_ylabel("Overall MAE (↓ better)", fontsize=10, fontfamily="DejaVu Sans")
ax_a.set_title("(a) Evidence prediction MAE\nacross 5 seeds", fontsize=10,
               fontfamily="DejaVu Sans", pad=6)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.set_xlim(-0.5, 2.8)
ax_a.tick_params(labelsize=9)

# ─── Panel (b): Per-target R² heatmap ─────────────────────────────────
r2_matrix = np.array([r2_data[v] for v in VARIANTS])  # shape 3 × 6

im = ax_b.imshow(r2_matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.6)
ax_b.set_xticks(np.arange(len(TARGETS)))
ax_b.set_xticklabels(TARGET_LABELS, fontsize=8, fontfamily="DejaVu Sans",
                     rotation=35, ha="right", rotation_mode="anchor")
ax_b.set_yticks(np.arange(len(VARIANTS)))
ax_b.set_yticklabels(VARIANT_LABELS, fontsize=9, fontfamily="DejaVu Sans")

for i in range(len(VARIANTS)):
    for j in range(len(TARGETS)):
        val = r2_matrix[i, j]
        txt_color = "white" if val > 0.4 else "black"
        ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                  fontsize=8.5, color=txt_color, fontfamily="DejaVu Sans")

cbar = fig.colorbar(im, ax=ax_b, shrink=0.85)
cbar.set_label("R² (5-seed mean)", fontsize=9, fontfamily="DejaVu Sans")
cbar.ax.tick_params(labelsize=8)
ax_b.set_title("(b) Per-target R² (5-seed mean)", fontsize=10,
               fontfamily="DejaVu Sans", pad=6)

plt.tight_layout(pad=1.5)
out_path = ("."
            "/figures/figure3_p2b_mae_r2.tiff")
fig.savefig(out_path, dpi=300, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {out_path}")
plt.close(fig)
