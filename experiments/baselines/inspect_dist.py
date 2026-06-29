"""Inspect the softmax max distribution across cases (for sanity check)."""
import numpy as np
bare = dict(np.load(r'./experiments/baselines/picai_val_bare_seed42.npz'))
labels = dict(np.load(r'./experiments/baselines/picai_val_labels.npz'))
pos_max, neg_max = [], []
for cid, det in bare.items():
    lbl = labels[cid]
    is_pos = bool((lbl > 0).any())
    m = float(np.squeeze(det).max())
    (pos_max if is_pos else neg_max).append(m)
print(f'Pos n={len(pos_max)} min={min(pos_max):.4f} p5={np.percentile(pos_max,5):.4f} p50={np.percentile(pos_max,50):.4f} max={max(pos_max):.4f}')
print(f'Neg n={len(neg_max)} min={min(neg_max):.4f} p50={np.percentile(neg_max,50):.4f} p75={np.percentile(neg_max,75):.4f} p95={np.percentile(neg_max,95):.4f} max={max(neg_max):.4f}')
# Hist of neg cases at high thresholds
import collections
print("Neg max histogram:")
bins = [0, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0]
for i in range(len(bins)-1):
    n = sum(1 for m in neg_max if bins[i] <= m < bins[i+1])
    print(f'  [{bins[i]}, {bins[i+1]}): {n}')
# And pos
print("Pos max histogram:")
for i in range(len(bins)-1):
    n = sum(1 for m in pos_max if bins[i] <= m < bins[i+1])
    print(f'  [{bins[i]}, {bins[i+1]}): {n}')

# Now the raw logit distribution
logits = dict(np.load(r'./experiments/baselines/picai_val_logits_bare_seed42.npz'))
pos_lgt, neg_lgt = [], []
for cid, lg in logits.items():
    lbl = labels[cid]
    is_pos = bool((lbl > 0).any())
    l0 = lg[0].astype(np.float32); l1 = lg[1].astype(np.float32)
    lgt = float((l1 - l0).max())
    (pos_lgt if is_pos else neg_lgt).append(lgt)
print(f'Pos raw_logit max: n={len(pos_lgt)} min={min(pos_lgt):.2f} p5={np.percentile(pos_lgt,5):.2f} p50={np.percentile(pos_lgt,50):.2f} max={max(pos_lgt):.2f}')
print(f'Neg raw_logit max: n={len(neg_lgt)} min={min(neg_lgt):.2f} p5={np.percentile(neg_lgt,5):.2f} p50={np.percentile(neg_lgt,50):.2f} p95={np.percentile(neg_lgt,95):.2f} max={max(neg_lgt):.2f}')
