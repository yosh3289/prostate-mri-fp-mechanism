#!/usr/bin/env python3
"""Case-level paired McNemar's test on Prostate158 (bare A2 vs P2a).

For each seed, at each model's matched-sensitivity threshold, binarizes the
per-case detection probability map (component-size >= min_voxels). Builds the
2x2 paired contingency (per case: bare_pos vs p2a_pos). Runs McNemar's test
with continuity correction (and exact binomial when discordant sum < 25).

Inputs
------
  experiments/prostate158_p2a_bare/detections_seed{S}.npz   (bare A2 per-case 3D maps)
  experiments/prostate158_p2a/detections_seed{S}.npz        (P2a per-case 3D maps)
  experiments/prostate158_matched_sens/matched_sens_seed{S}.json  (tau_match_bare, tau_match_p2a)
  Prostate158 cache (per-case labels)

Outputs
-------
  experiments/prostate158_casewise/casewise_mcnemar_seed{S}.json
  experiments/prostate158_casewise/casewise_mcnemar_aggregate.json

Statistical framework
---------------------
  Statistical unit = CASE (n=158). Seed is a replicate, not a sample.
  For each seed we report a McNemar p-value on the paired 2x2 table
  constructed from the 158 per-case binary predictions. For the pooled
  estimate across seeds we sum discordant cells and apply an exact
  binomial two-sided test on b vs c (total = b + c successes split 50/50
  under H0), which is the standard small-count substitute for McNemar.

Usage
-----
  conda run -n mar python experiments/prostate158_casewise/casewise_mcnemar.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
P2A_DIR = EXPERIMENTS_DIR / "prostate158_p2a"
BARE_DIR = EXPERIMENTS_DIR / "prostate158_p2a_bare"
MATCHED_DIR = EXPERIMENTS_DIR / "prostate158_matched_sens"
P158_CACHE = Path(
    "ADJUST_PATH/prostate158_cache  (NOT in repo: local Prostate158 preprocessed cache)"
)
OUT_DIR = Path(__file__).resolve().parent

SEEDS = [42, 123, 456, 789, 1024]
MIN_VOXELS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_gt_labels() -> dict:
    """Return {pid: is_positive(bool)} for Prostate158."""
    out = {}
    for f in sorted(P158_CACHE.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        pid = d["patient_id"]
        label = d["label"]
        arr = label.numpy() if hasattr(label, "numpy") else np.asarray(label)
        while arr.ndim > 3:
            arr = arr[0]
        out[pid] = bool((arr > 0.5).any())
    return out


def binarize_case(det: np.ndarray, threshold: float, min_voxels: int = MIN_VOXELS) -> bool:
    """Mirror compute_matched_sens.compute_casesens_casespec binarization rule."""
    binary = det > threshold
    if not binary.any():
        return False
    labeled, n = ndimage.label(binary)
    if n == 0:
        return False
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return bool(any(s >= min_voxels for s in sizes))


def mcnemar_stat(b: int, c: int) -> tuple[float, float, str]:
    """McNemar with continuity correction; falls back to exact binomial when b+c < 25.

    Returns (stat, p_value, method).
    """
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0, "degenerate_no_discordance"
    if n_disc < 25:
        # Exact two-sided binomial (H0: Pr(discordant favors bare) = 0.5).
        # Use scipy.stats.binomtest (scipy >= 1.7).
        try:
            res = stats.binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided")
            p = float(res.pvalue)
        except AttributeError:  # older scipy
            res = stats.binom_test(min(b, c), n_disc, p=0.5, alternative="two-sided")
            p = float(res)
        return float(n_disc), p, "exact_binomial_two_sided"
    # Standard McNemar with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n_disc
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p), "mcnemar_continuity_corrected"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading Prostate158 ground-truth labels from %s ...", P158_CACHE)
    gt = load_gt_labels()
    n_pos = sum(1 for v in gt.values() if v)
    log.info("Loaded GT for %d cases (%d positive, %d negative)", len(gt), n_pos, len(gt) - n_pos)

    per_seed_results = []
    pooled_b = 0  # bare-pos, P2a-neg
    pooled_c = 0  # bare-neg, P2a-pos

    for seed in SEEDS:
        bare_path = BARE_DIR / f"detections_seed{seed}.npz"
        p2a_path = P2A_DIR / f"detections_seed{seed}.npz"
        matched_path = MATCHED_DIR / f"matched_sens_seed{seed}.json"
        for p in (bare_path, p2a_path, matched_path):
            if not p.exists():
                raise FileNotFoundError(p)

        matched = json.loads(matched_path.read_text())
        tau_bare = float(matched["tau_match_bare"])
        tau_p2a = float(matched["tau_match_p2a"])
        log.info("=== Seed %d  (tau_bare=%.2f  tau_p2a=%.2f) ===", seed, tau_bare, tau_p2a)

        bare_det = dict(np.load(bare_path))
        p2a_det = dict(np.load(p2a_path))

        # Use common case ordering
        pids = sorted(set(bare_det.keys()) & set(p2a_det.keys()) & set(gt.keys()))
        n = len(pids)
        log.info("  common cases: %d", n)

        # 2x2 contingency: rows = bare (pos/neg), cols = P2a (pos/neg)
        a = b = c = d = 0  # a: both pos; b: bare pos, p2a neg; c: bare neg, p2a pos; d: both neg
        # Also track by GT:
        bare_only_pos_gtpos = bare_only_pos_gtneg = 0
        p2a_only_pos_gtpos = p2a_only_pos_gtneg = 0

        for pid in pids:
            bare_pred = binarize_case(bare_det[pid], tau_bare)
            p2a_pred = binarize_case(p2a_det[pid], tau_p2a)
            is_pos = gt[pid]

            if bare_pred and p2a_pred:
                a += 1
            elif bare_pred and not p2a_pred:
                b += 1
                if is_pos:
                    bare_only_pos_gtpos += 1
                else:
                    bare_only_pos_gtneg += 1
            elif not bare_pred and p2a_pred:
                c += 1
                if is_pos:
                    p2a_only_pos_gtpos += 1
                else:
                    p2a_only_pos_gtneg += 1
            else:
                d += 1

        stat, p_val, method = mcnemar_stat(b, c)
        log.info(
            "  contingency: a=%d (both+), b=%d (bare+ p2a-), c=%d (bare- p2a+), d=%d (both-)",
            a, b, c, d,
        )
        log.info("  McNemar: stat=%.4f  p=%.4f  method=%s", stat, p_val, method)
        log.info("  bare-only-flagged: GT+=%d, GT-=%d", bare_only_pos_gtpos, bare_only_pos_gtneg)
        log.info("  p2a-only-flagged : GT+=%d, GT-=%d", p2a_only_pos_gtpos, p2a_only_pos_gtneg)

        seed_result = {
            "seed": seed,
            "tau_match_bare": tau_bare,
            "tau_match_p2a": tau_p2a,
            "min_voxels": MIN_VOXELS,
            "n_cases": n,
            "contingency": {
                "both_positive": a,
                "bare_pos_p2a_neg": b,
                "bare_neg_p2a_pos": c,
                "both_negative": d,
            },
            "bare_only_positive": {"gt_positive": bare_only_pos_gtpos, "gt_negative": bare_only_pos_gtneg},
            "p2a_only_positive": {"gt_positive": p2a_only_pos_gtpos, "gt_negative": p2a_only_pos_gtneg},
            "mcnemar": {
                "statistic": stat,
                "p_value": p_val,
                "method": method,
                "alternative": "two_sided",
                "note": "Continuity-corrected McNemar (chi2, df=1) for b+c >= 25; exact binomial otherwise",
            },
        }
        out_path = OUT_DIR / f"casewise_mcnemar_seed{seed}.json"
        out_path.write_text(json.dumps(seed_result, indent=2))
        log.info("  Written: %s", out_path)
        per_seed_results.append(seed_result)

        pooled_b += b
        pooled_c += c

    # Pooled across seeds: sum discordant cells, exact binomial on b vs c.
    pooled_stat, pooled_p, pooled_method = mcnemar_stat(pooled_b, pooled_c)
    log.info("=== POOLED ACROSS %d SEEDS ===", len(per_seed_results))
    log.info(
        "  pooled b=%d (bare+ p2a-)  c=%d (bare- p2a+)  -> stat=%.4f  p=%.4f  method=%s",
        pooled_b, pooled_c, pooled_stat, pooled_p, pooled_method,
    )

    per_seed_p = [r["mcnemar"]["p_value"] for r in per_seed_results]

    agg = {
        "n_seeds": len(per_seed_results),
        "seeds": [r["seed"] for r in per_seed_results],
        "min_voxels": MIN_VOXELS,
        "n_cases_per_seed": [r["n_cases"] for r in per_seed_results],
        "per_seed_contingency": {r["seed"]: r["contingency"] for r in per_seed_results},
        "per_seed_mcnemar_p": {r["seed"]: r["mcnemar"]["p_value"] for r in per_seed_results},
        "per_seed_mcnemar_method": {r["seed"]: r["mcnemar"]["method"] for r in per_seed_results},
        "pooled_discordant": {"b_bare_pos_p2a_neg": pooled_b, "c_bare_neg_p2a_pos": pooled_c},
        "pooled_mcnemar": {
            "statistic": pooled_stat,
            "p_value": pooled_p,
            "method": pooled_method,
            "alternative": "two_sided",
            "note": "Exact binomial two-sided on min(b,c)|b+c; standard small-count McNemar substitute.",
        },
        "min_per_seed_p": float(min(per_seed_p)),
        "max_per_seed_p": float(max(per_seed_p)),
    }
    agg_path = OUT_DIR / "casewise_mcnemar_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2))
    log.info("Written aggregate: %s", agg_path)


if __name__ == "__main__":
    main()
