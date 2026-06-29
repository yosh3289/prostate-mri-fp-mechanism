#!/usr/bin/env python3
"""Per-case bootstrap 95% CI for paired Delta CaseSpec (P2a - bare) at matched sensitivity.

For each seed, at each model's matched-sensitivity threshold, computes a
per-case specificity contribution (1 if negative-case and correctly predicted
negative, 0 otherwise) for negative cases only. Paired per-case differences
are resampled with replacement (10 000 iterations) to obtain the
bootstrap 95% CI on the mean paired difference. A two-sided bootstrap p-value
is computed via central percentile inversion around H0: mean = 0.

PI-CAI per-case note
--------------------
Per-case P2a detection sidecars for PI-CAI fold-0 val are not present in the
local experiments tree (only aggregate scenario JSONs exist). This analysis
therefore covers the Prostate158 external cohort only; the PI-CAI extension is
blocked on per-case sidecar availability.

Inputs
------
  experiments/prostate158_p2a_bare/detections_seed{S}.npz
  experiments/prostate158_p2a/detections_seed{S}.npz
  experiments/prostate158_matched_sens/matched_sens_seed{S}.json
  Prostate158 cache (labels)

Outputs
-------
  experiments/casewise_bootstrap/bootstrap_ci_seed{S}.json
  experiments/casewise_bootstrap/bootstrap_ci_aggregate.json

Usage
-----
  conda run -n mar python experiments/casewise_bootstrap/bootstrap_ci.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
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
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260421  # reproducible
MIN_VOXELS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_gt_labels() -> dict:
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
    binary = det > threshold
    if not binary.any():
        return False
    labeled, n = ndimage.label(binary)
    if n == 0:
        return False
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return bool(any(s >= min_voxels for s in sizes))


def paired_bootstrap(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Classic paired bootstrap: resample cases with replacement, compute mean of paired diffs.

    Returns dict with mean, 95% CI (2.5/97.5 percentiles), and a two-sided
    bootstrap p-value (proportion of resamples whose mean has opposite sign of
    the observed mean, doubled).
    """
    n = len(diffs)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    observed_mean = float(diffs.mean())
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    # Two-sided p via percentile inversion:
    # p = 2 * min(Pr(boot <= 0), Pr(boot >= 0))
    prop_le_zero = float((boot_means <= 0).mean())
    prop_ge_zero = float((boot_means >= 0).mean())
    p_two_sided = float(min(1.0, 2.0 * min(prop_le_zero, prop_ge_zero)))
    return {
        "observed_mean_paired_diff": observed_mean,
        "ci_2p5": lo,
        "ci_97p5": hi,
        "p_two_sided": p_two_sided,
        "n_cases": n,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading Prostate158 ground-truth labels from %s ...", P158_CACHE)
    gt = load_gt_labels()
    n_pos = sum(1 for v in gt.values() if v)
    log.info("Loaded GT for %d cases (%d positive, %d negative)", len(gt), n_pos, len(gt) - n_pos)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    per_seed_results = []
    # Pooled: concatenate per-case diffs across seeds (treating each (seed, negative case)
    # as an observation with case clustered). Primary inference is per-seed + aggregate.

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
        pids = sorted(set(bare_det.keys()) & set(p2a_det.keys()) & set(gt.keys()))

        # Per-case specificity contribution for negative cases only.
        # spec_contrib = 1 if predicted-negative, 0 otherwise.
        diffs = []
        for pid in pids:
            if gt[pid]:  # positive case -> not contributing to specificity
                continue
            bare_pred = binarize_case(bare_det[pid], tau_bare)
            p2a_pred = binarize_case(p2a_det[pid], tau_p2a)
            bare_spec = 0 if bare_pred else 1
            p2a_spec = 0 if p2a_pred else 1
            diffs.append(p2a_spec - bare_spec)
        diffs = np.asarray(diffs, dtype=np.float64)
        log.info("  negatives contributing to spec: %d", len(diffs))

        boot = paired_bootstrap(diffs, N_BOOTSTRAP, rng)
        log.info(
            "  mean paired diff (P2a-bare) = %+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f",
            boot["observed_mean_paired_diff"], boot["ci_2p5"], boot["ci_97p5"], boot["p_two_sided"],
        )

        seed_result = {
            "seed": seed,
            "cohort": "prostate158",
            "tau_match_bare": tau_bare,
            "tau_match_p2a": tau_p2a,
            "min_voxels": MIN_VOXELS,
            "n_negative_cases": int(len(diffs)),
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_rng_seed": BOOTSTRAP_SEED,
            "bootstrap": boot,
        }
        out_path = OUT_DIR / f"bootstrap_ci_seed{seed}.json"
        out_path.write_text(json.dumps(seed_result, indent=2))
        log.info("  Written: %s", out_path)
        per_seed_results.append(seed_result)

    # Aggregate: mean of per-seed point estimates; pooled bootstrap over (seed, case) pairs.
    per_seed_means = np.array([r["bootstrap"]["observed_mean_paired_diff"] for r in per_seed_results])
    per_seed_lo = np.array([r["bootstrap"]["ci_2p5"] for r in per_seed_results])
    per_seed_hi = np.array([r["bootstrap"]["ci_97p5"] for r in per_seed_results])
    per_seed_p = [r["bootstrap"]["p_two_sided"] for r in per_seed_results]

    # Pooled: concatenate per-case paired diffs across seeds, bootstrap again.
    all_diffs = []
    for seed in SEEDS:
        matched = json.loads((MATCHED_DIR / f"matched_sens_seed{seed}.json").read_text())
        tau_bare = float(matched["tau_match_bare"])
        tau_p2a = float(matched["tau_match_p2a"])
        bare_det = dict(np.load(BARE_DIR / f"detections_seed{seed}.npz"))
        p2a_det = dict(np.load(P2A_DIR / f"detections_seed{seed}.npz"))
        for pid in sorted(set(bare_det.keys()) & set(p2a_det.keys()) & set(gt.keys())):
            if gt[pid]:
                continue
            bare_pred = binarize_case(bare_det[pid], tau_bare)
            p2a_pred = binarize_case(p2a_det[pid], tau_p2a)
            all_diffs.append((0 if p2a_pred else 1) - (0 if bare_pred else 1))
    pooled_diffs = np.asarray(all_diffs, dtype=np.float64)
    pooled_boot = paired_bootstrap(pooled_diffs, N_BOOTSTRAP, rng)
    log.info(
        "=== POOLED (all seeds, %d case-observations) mean=%+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f",
        len(pooled_diffs), pooled_boot["observed_mean_paired_diff"],
        pooled_boot["ci_2p5"], pooled_boot["ci_97p5"], pooled_boot["p_two_sided"],
    )

    agg = {
        "cohort": "prostate158",
        "n_seeds": len(per_seed_results),
        "seeds": SEEDS,
        "n_bootstrap": N_BOOTSTRAP,
        "per_seed_mean": per_seed_means.tolist(),
        "per_seed_ci_2p5": per_seed_lo.tolist(),
        "per_seed_ci_97p5": per_seed_hi.tolist(),
        "per_seed_p_two_sided": per_seed_p,
        "mean_of_per_seed_means": float(per_seed_means.mean()),
        "sd_of_per_seed_means": float(per_seed_means.std(ddof=1)),
        "pooled_cases_bootstrap": {
            "n_case_observations": int(len(pooled_diffs)),
            "note": (
                "Pooled across 5 seeds: each (seed, negative case) treated as one observation; "
                "paired diff P2a_spec - bare_spec. Bootstrap resamples the pooled observation index."
            ),
            **pooled_boot,
        },
        "picai_note": (
            "PI-CAI per-case detection sidecars are not present in the local experiments tree "
            "(eval JSONs contain only aggregate scenario scalars). PI-CAI bootstrap is therefore "
            "blocked on per-case sidecars; Prostate158 analysis is reported."
        ),
    }
    agg_path = OUT_DIR / "bootstrap_ci_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2))
    log.info("Written aggregate: %s", agg_path)


if __name__ == "__main__":
    main()
