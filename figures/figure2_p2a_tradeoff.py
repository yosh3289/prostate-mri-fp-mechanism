"""
Figure 2 — P2a tradeoff (2 panels)
  (a) Per-seed CaseSens vs CaseSpec scatter on PI-CAI (paired seeds; existing)
  (b) Per-case paired scatter on Prostate158: max probability at matched-sens
      threshold for bare A2 vs P2a, colour-coded by GT (cancer vs negative).
      Visual message: how many cases cross the decision boundary differently
      between the two models.
EurRadiol Paper 2, Session 7
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Per-seed data on PI-CAI (Panel a) ──────────────────────────────────
# Bare A2: seed → (CaseSens, CaseSpec)
BARE = {
    42:   (0.9405, 0.5787),
    123:  (0.9405, 0.2546),
    456:  (0.9405, 0.6759),
    789:  (0.9762, 0.3056),
    1024: (0.9167, 0.5278),
}
# P2a ms v2: seed → (CaseSens, CaseSpec)
P2A = {
    42:   (0.9405, 0.6296),
    123:  (0.9405, 0.3380),
    456:  (0.9405, 0.6852),
    789:  (0.9762, 0.5463),
    1024: (0.9167, 0.5463),
}
SEEDS = [42, 123, 456, 789, 1024]
SEED_LABELS = {42: "S42", 123: "S123", 456: "S456", 789: "S789", 1024: "S1024"}

bare_sens = np.array([BARE[s][0] for s in SEEDS])
bare_spec = np.array([BARE[s][1] for s in SEEDS])
p2a_sens  = np.array([P2A[s][0]  for s in SEEDS])
p2a_spec  = np.array([P2A[s][1]  for s in SEEDS])

bare_mean = (np.mean(bare_sens), np.mean(bare_spec))
bare_sd   = (np.std(bare_sens, ddof=1), np.std(bare_spec, ddof=1))
p2a_mean  = (np.mean(p2a_sens), np.mean(p2a_spec))
p2a_sd    = (np.std(p2a_sens, ddof=1), np.std(p2a_spec, ddof=1))

COLOR_BARE = "#888888"
COLOR_P2A  = "#1f77b4"
COLOR_POS  = "#2CA02C"  # green  = GT cancer
COLOR_NEG  = "#9A9A9A"  # grey   = GT negative

# ── Prostate158 per-case data (Panel b) ────────────────────────────────
PROBS_PATH = Path(
    "./figures/prostate158_casewise_probs.json"
)
probs = json.loads(PROBS_PATH.read_text())
# Use seed 1024 (largest positive delta for bare; most informative panel).
# We also pool the other seeds for the scatter to make the "how many cross
# the boundary" count visible; pooled across all 5 seeds × 158 cases = 790
# observations, but alpha-blended so the plot is not saturated.
PANEL_B_SEED = 1024  # primary seed shown clearly
entry = probs[f"seed{PANEL_B_SEED}"]
tau_b = entry["tau_match_bare"]
tau_p = entry["tau_match_p2a"]
bare_max = np.array(entry["bare_max"], dtype=float)
p2a_max  = np.array(entry["p2a_max"],  dtype=float)
gt_arr   = np.array(entry["gt"],       dtype=bool)
n_cases  = len(bare_max)

# Count how many cases cross the decision boundary differently:
bare_pos = bare_max > tau_b
p2a_pos  = p2a_max  > tau_p
bare_only = int(((bare_pos) & (~p2a_pos)).sum())
p2a_only  = int(((~bare_pos) & (p2a_pos)).sum())
both_pos  = int(((bare_pos) & (p2a_pos)).sum())
both_neg  = int(((~bare_pos) & (~p2a_pos)).sum())

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ══════════════ Panel (a): PI-CAI per-seed scatter ══════════════
ax = axes[0]
for s in SEEDS:
    xs = [BARE[s][0], P2A[s][0]]
    ys = [BARE[s][1], P2A[s][1]]
    ax.plot(xs, ys, color="#BBBBBB", linewidth=0.9, zorder=1)

ax.scatter(bare_sens, bare_spec, color=COLOR_BARE, marker="o", s=55,
           label="Bare A2", zorder=3, edgecolors="white", linewidths=0.5)
ax.scatter(p2a_sens, p2a_spec, color=COLOR_P2A, marker="o", s=55,
           label="P2a (ms v2)", zorder=3, edgecolors="white", linewidths=0.5)
# Manual per-seed label offsets (points) + leader lines to avoid mutual
# occlusion — seeds 42/123/456 share sensitivity 0.9405 (same x column).
SEED_OFFSETS = {
    42:   (10, 9, "left"),
    123:  (10, -14, "left"),
    456:  (-10, 10, "right"),
    789:  (12, 2, "left"),
    1024: (-12, 2, "right"),
}
for s in SEEDS:
    x, y = P2A[s]
    dx, dy, ha = SEED_OFFSETS[s]
    ax.annotate(SEED_LABELS[s], xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, color=COLOR_P2A,
                fontfamily="DejaVu Sans", ha=ha, va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color=COLOR_P2A,
                                linewidth=0.5, alpha=0.6,
                                shrinkA=0, shrinkB=3))
ax.errorbar(bare_mean[0], bare_mean[1],
            xerr=bare_sd[0], yerr=bare_sd[1],
            fmt="D", color=COLOR_BARE, markersize=10, capsize=4,
            linewidth=1.5, zorder=4, label="Bare mean ± SD")
ax.errorbar(p2a_mean[0], p2a_mean[1],
            xerr=p2a_sd[0], yerr=p2a_sd[1],
            fmt="D", color=COLOR_P2A, markersize=10, capsize=4,
            linewidth=1.5, zorder=4, label="P2a mean ± SD")
ann_text = (
    f"Bare A2 : Sens {bare_mean[0]:.3f} ± {bare_sd[0]:.3f}\n"
    f"          Spec {bare_mean[1]:.3f} ± {bare_sd[1]:.3f}\n"
    f"P2a ms v2: Sens {p2a_mean[0]:.3f} ± {p2a_sd[0]:.3f}\n"
    f"          Spec {p2a_mean[1]:.3f} ± {p2a_sd[1]:.3f}"
)
ax.text(0.02, 0.02, ann_text, transform=ax.transAxes,
        fontsize=7.5, va="bottom", ha="left", fontfamily="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#AAAAAA", alpha=0.9))
ax.set_xlabel("Case-level sensitivity", fontsize=10, fontfamily="DejaVu Sans")
ax.set_ylabel("Case-level specificity", fontsize=10, fontfamily="DejaVu Sans")
ax.set_xlim(0.85, 1.01)
ax.set_ylim(-0.02, 0.85)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Clean custom legend (avoid the messy errorbar legend handler)
from matplotlib.lines import Line2D
legend_handles_a = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BARE,
           markeredgecolor="white", markersize=8, label="Bare A2 (per seed)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_P2A,
           markeredgecolor="white", markersize=8, label="P2a ms v2 (per seed)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="#CCCCCC",
           markeredgecolor="#222222", markersize=9, label="Mean ± SD"),
]
ax.legend(handles=legend_handles_a, loc="lower right", fontsize=8, framealpha=0.92,
          prop={"family": "DejaVu Sans"}, borderpad=0.6, handletextpad=0.5)
ax.set_title("(a)  PI-CAI fold-0 validation (5 seeds, paired)",
             fontsize=10, fontfamily="DejaVu Sans", loc="left")

# ══════════════ Panel (b): Prostate158 per-case paired scatter ══════════════
# Per-case max probabilities saturate (100% > 0.9, 94% > 0.99), so a linear
# axis piles every case in the top-right corner. We plot on a LOGIT-SPACED
# axis (spreads the 0.95–1.0 region) but label ticks in PROBABILITY units so
# the operating points remain clinically readable. Small jitter separates
# near-identical values.
ax = axes[1]

def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

# Probability tick marks shown on both axes
PROB_TICKS = [0.90, 0.96, 0.99, 0.999, 0.9999]
TICK_POS = logit(PROB_TICKS)
TICK_LBL = ["0.90", "0.96", "0.99", "0.999", "0.9999"]

rng_b = np.random.default_rng(2024)
jit = lambda n: rng_b.uniform(-0.06, 0.06, n)

bx = logit(bare_max); by = logit(p2a_max)
lo_l, hi_l = logit(0.90), logit(0.9999)

# Identity line + threshold guides in logit space
ax.plot([lo_l, hi_l], [lo_l, hi_l], linestyle="--", color="#999999",
        linewidth=1.0, zorder=1, label="y = x")
tb_l, tp_l = logit(tau_b), logit(tau_p)
ax.axvline(tb_l, color=COLOR_BARE, linestyle=":", linewidth=1.0, zorder=1, alpha=0.8)
ax.axhline(tp_l, color=COLOR_P2A, linestyle=":", linewidth=1.0, zorder=1, alpha=0.8)
ax.text(tb_l, lo_l + 0.15, f" τ_bare={tau_b:.2f}", fontsize=7.5,
        color=COLOR_BARE, ha="left", va="bottom", fontfamily="DejaVu Sans", rotation=90)
ax.text(lo_l + 0.1, tp_l, f" τ_p2a={tau_p:.2f}", fontsize=7.5,
        color=COLOR_P2A, ha="left", va="bottom", fontfamily="DejaVu Sans")

pos_mask = gt_arr
ax.scatter(bx[~pos_mask] + jit((~pos_mask).sum()), by[~pos_mask] + jit((~pos_mask).sum()),
           c=COLOR_NEG, marker="o", s=30, alpha=0.75,
           edgecolors="white", linewidths=0.4, zorder=3,
           label=f"GT negative (n = {(~pos_mask).sum()})")
ax.scatter(bx[pos_mask] + jit(pos_mask.sum()), by[pos_mask] + jit(pos_mask.sum()),
           c=COLOR_POS, marker="^", s=34, alpha=0.85,
           edgecolors="white", linewidths=0.4, zorder=4,
           label=f"GT cancer (n = {pos_mask.sum()})")

b_ann = (
    f"Seed {PANEL_B_SEED}, n = {n_cases} cases\n"
    f"Both flag positive : {both_pos}\n"
    f"Both flag negative : {both_neg}\n"
    f"Bare only flags +  : {bare_only}\n"
    f"P2a only flags +   : {p2a_only}"
)
ax.text(0.98, 0.02, b_ann, transform=ax.transAxes,
        fontsize=7.5, va="bottom", ha="right", fontfamily="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#AAAAAA", alpha=0.92))

ax.set_xlabel("Bare A2 per-case max probability (logit-spaced)",
              fontsize=10, fontfamily="DejaVu Sans")
ax.set_ylabel("P2a per-case max probability (logit-spaced)",
              fontsize=10, fontfamily="DejaVu Sans")
ax.set_xticks(TICK_POS); ax.set_xticklabels(TICK_LBL)
ax.set_yticks(TICK_POS); ax.set_yticklabels(TICK_LBL)
ax.set_xlim(lo_l - 0.2, hi_l + 0.3)
ax.set_ylim(lo_l - 0.2, hi_l + 0.3)
ax.tick_params(labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
          prop={"family": "DejaVu Sans"})
ax.set_title("(b)  Prostate158 per-case paired probabilities at matched-sens τ",
             fontsize=10, fontfamily="DejaVu Sans", loc="left")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

out_path = "./figures/figure2_p2a_tradeoff.tiff"
fig.savefig(out_path, dpi=300, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {out_path}")
print(f"Panel (a) Bare  mean sens={bare_mean[0]:.4f} sd={bare_sd[0]:.4f}, "
      f"spec={bare_mean[1]:.4f} sd={bare_sd[1]:.4f}")
print(f"Panel (a) P2a   mean sens={p2a_mean[0]:.4f} sd={p2a_sd[0]:.4f}, "
      f"spec={p2a_mean[1]:.4f} sd={p2a_sd[1]:.4f}")
print(f"Panel (b) seed={PANEL_B_SEED} n={n_cases}; "
      f"both+={both_pos}, both-={both_neg}, bare_only+={bare_only}, "
      f"p2a_only+={p2a_only}")
plt.close(fig)
