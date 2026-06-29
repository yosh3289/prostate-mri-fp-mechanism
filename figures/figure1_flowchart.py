"""
Figure 1 — Study flowchart (static matplotlib patch diagram)
EurRadiol Paper 2, Session 3
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(2.05, 8.15)   # crop empty lower half; content lives in 2.5–7.85
ax.axis("off")

# Colors
BOX_FACE = "#E8E8E8"
BOX_EDGE = "#404040"
ARROW_COLOR = "#404040"
TEXT_COLOR = "#000000"

def draw_box(ax, cx, cy, w, h, text, fontsize=9):
    x0 = cx - w / 2
    y0 = cy - h / 2
    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.05",
        facecolor=BOX_FACE,
        edgecolor=BOX_EDGE,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        cx, cy, text,
        ha="center", va="center",
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontfamily="DejaVu Sans",
        zorder=4,
        wrap=True,
    )

def draw_arrow(ax, x0, y0, x1, y1, label=None, color=ARROW_COLOR, style="solid"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.18",
            color=color,
            linewidth=1.5,
            linestyle=style,
        ),
        zorder=5,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.1, my, label, fontsize=8, color=color, ha="left", va="center",
                fontfamily="DejaVu Sans", zorder=6)

# ── Column A: PI-CAI ────────────────────────────────────────────────────
col_a = 2.5
box_w = 3.8
box_h = 0.72

y_a = [7.0, 5.6, 4.1, 2.5]
labels_a = [
    "PI-CAI\n(n = 1500)",
    "Fold-0 split:\n1200 train / 300 val",
    "Refinement head trained\n(5 seeds)",
    "Primary evaluation:\nfold-0 validation (n = 300)",
]

for y, lbl in zip(y_a, labels_a):
    draw_box(ax, col_a, y, box_w, box_h, lbl)

for ya, yb in zip(y_a[:-1], y_a[1:]):
    draw_arrow(ax, col_a, ya - box_h / 2, col_a, yb + box_h / 2)

# Column header
ax.text(col_a, 7.78, "Column A — PI-CAI (primary cohort)", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#222222", fontfamily="DejaVu Sans")

# ── Column B: Prostate158 ───────────────────────────────────────────────
col_b = 7.5
y_b = [7.0, 5.6, 4.1]
labels_b = [
    "Prostate158\n(n = 158)",
    "Same preprocessing\npipeline",
    "External evaluation\n(5 seeds, no retraining)",
]

for y, lbl in zip(y_b, labels_b):
    draw_box(ax, col_b, y, box_w, box_h, lbl)

for ya, yb in zip(y_b[:-1], y_b[1:]):
    draw_arrow(ax, col_b, ya - box_h / 2, col_b, yb + box_h / 2)

ax.text(col_b, 7.78, "Column B — Prostate158 (external cohort)", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#222222", fontfamily="DejaVu Sans")

# ── Horizontal "weight transfer" arrow: col_a y=4.1  →  col_b y=4.1 ──
ax.annotate(
    "",
    xy=(col_b - box_w / 2, 4.1),
    xytext=(col_a + box_w / 2, 4.1),
    arrowprops=dict(
        arrowstyle="->,head_width=0.28,head_length=0.22",
        color="#2060A0",
        linewidth=1.8,
        linestyle="dashed",
        connectionstyle="arc3,rad=0.0",
    ),
    zorder=5,
)
ax.text(5.0, 4.35, "Weight transfer\n(frozen weights)", ha="center", va="bottom",
        fontsize=8, color="#2060A0", fontfamily="DejaVu Sans", fontstyle="italic", zorder=6)

plt.tight_layout(pad=0.4)
out_path = "./figures/figure1_flowchart.tiff"
fig.savefig(out_path, dpi=300, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print(f"Saved: {out_path}")
plt.close(fig)
