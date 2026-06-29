#!/usr/bin/env python3
"""Bootstrap-CI over PI-CAI fold-0 val cases (alternative to 5-fold CV).

Since fold-1/2/3/4 backbone checkpoints are not available (see BLOCKED.md),
this script computes a non-parametric bootstrap 95% CI over case-level
CaseSens / CaseSpec / delta-CaseSpec (P2a - bare) for each seed, using
B=1000 bootstrap resamples of the 300 PI-CAI val cases.

For each seed:
  - Load bare detections and P2a detections (per-case softmax maps)
  - Load GT labels
  - Bootstrap-resample case indices B times
  - At the matched-sens threshold (target CaseSens = 0.94), compute
    CaseSpec for bare and P2a on each resample
  - Report mean, SD, 95% percentile CI

Outputs:
  experiments/p2a_5fold_cv/bootstrap_seed{S}.json
  experiments/p2a_5fold_cv/bootstrap_results.json  (aggregate)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage

EXPERIMENTS = Path("./experiments")
PICAI_BARE = EXPERIMENTS / "baselines" / "picai_val_bare_seed{seed}.npz"
PICAI_P2A = EXPERIMENTS / "baselines" / "picai_val_p2a_seed{seed}.npz"
PICAI_LABELS = EXPERIMENTS / "baselines" / "picai_val_labels.npz"
OUT_DIR = EXPERIMENTS / "p2a_5fold_cv"

SEEDS = [42, 123, 456, 789, 1024]
TARGET_SENS = 0.94
MIN_VOXELS = 10
N_BOOTSTRAP = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def case_pred_positive(det_map: np.ndarray, threshold: float, min_voxels: int = MIN_VOXELS) -> bool:
    binary = det_map > threshold
    if not binary.any():
        return False
    labeled, n = ndimage.label(binary)
    if n == 0:
        return False
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return bool(any(s >= min_voxels for s in sizes))


def precompute_case_preds(detections: dict[str, np.ndarray], thresholds: np.ndarray,
                          min_voxels: int = MIN_VOXELS) -> dict[str, np.ndarray]:
    """For each case, return bool array of len(thresholds): predicted positive?"""
    out = {}
    for cid, det in detections.items():
        det = np.squeeze(det)
        preds = np.zeros(len(thresholds), dtype=bool)
        for i, t in enumerate(thresholds):
            preds[i] = case_pred_positive(det, float(t), min_voxels)
        out[cid] = preds
    return out


def compute_sens_spec_from_preds(preds_all: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """preds_all: [N_cases, N_thresholds] bool
    y_true: [N_cases] bool
    Returns (sens, spec) each of length N_thresholds.
    """
    pos_mask = y_true.astype(bool)
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    tp = preds_all[pos_mask].sum(axis=0)
    fn = n_pos - tp
    fp = preds_all[neg_mask].sum(axis=0)
    tn = n_neg - fp
    sens = tp / (tp + fn) if n_pos > 0 else np.full(preds_all.shape[1], np.nan)
    spec = tn / (tn + fp) if n_neg > 0 else np.full(preds_all.shape[1], np.nan)
    return sens, spec


def find_matched_tau_idx(sens: np.ndarray, spec: np.ndarray, target: float) -> int:
    """Find index of threshold minimising |sens - target|, tie-break by higher spec."""
    valid = np.isfinite(sens) & np.isfinite(spec)
    if not valid.any():
        return -1
    diff = np.abs(sens - target)
    diff[~valid] = np.inf
    min_diff = diff.min()
    candidates = np.where(np.abs(diff - min_diff) < 1e-9)[0]
    # Among candidates, pick highest spec
    best_idx = candidates[np.argmax(spec[candidates])]
    return int(best_idx)


def bootstrap_one_seed(seed: int, n_boot: int = N_BOOTSTRAP) -> dict:
    bare_path = Path(str(PICAI_BARE).format(seed=seed))
    p2a_path = Path(str(PICAI_P2A).format(seed=seed))
    if not (bare_path.exists() and p2a_path.exists() and PICAI_LABELS.exists()):
        raise FileNotFoundError(f"Missing inputs for seed {seed}")

    log.info("[seed=%d] loading detections and labels", seed)
    bare_det = dict(np.load(bare_path))
    p2a_det = dict(np.load(p2a_path))
    labels = dict(np.load(PICAI_LABELS))

    # Align case IDs
    common_ids = sorted(set(bare_det.keys()) & set(p2a_det.keys()) & set(labels.keys()))
    log.info("[seed=%d] %d common cases", seed, len(common_ids))

    y_true = np.array([bool((labels[c] > 0).any()) for c in common_ids])

    thresholds = np.arange(0.01, 1.00, 0.01)
    log.info("[seed=%d] precomputing per-case predictions across %d thresholds", seed, len(thresholds))
    bare_preds_dict = precompute_case_preds({c: bare_det[c] for c in common_ids}, thresholds)
    p2a_preds_dict = precompute_case_preds({c: p2a_det[c] for c in common_ids}, thresholds)

    bare_preds = np.stack([bare_preds_dict[c] for c in common_ids])  # [N, T]
    p2a_preds = np.stack([p2a_preds_dict[c] for c in common_ids])    # [N, T]

    # Full-cohort baseline: find matched tau for bare and P2a on full set
    bare_sens, bare_spec = compute_sens_spec_from_preds(bare_preds, y_true)
    p2a_sens, p2a_spec = compute_sens_spec_from_preds(p2a_preds, y_true)
    bare_idx = find_matched_tau_idx(bare_sens, bare_spec, TARGET_SENS)
    p2a_idx = find_matched_tau_idx(p2a_sens, p2a_spec, TARGET_SENS)
    full = {
        "bare_tau": float(thresholds[bare_idx]),
        "bare_casesens": float(bare_sens[bare_idx]),
        "bare_casespec": float(bare_spec[bare_idx]),
        "p2a_tau": float(thresholds[p2a_idx]),
        "p2a_casesens": float(p2a_sens[p2a_idx]),
        "p2a_casespec": float(p2a_spec[p2a_idx]),
        "delta_casespec": float(p2a_spec[p2a_idx] - bare_spec[bare_idx]),
    }

    # Bootstrap
    log.info("[seed=%d] running %d bootstrap resamples", seed, n_boot)
    rng = np.random.default_rng(seed)
    N = len(common_ids)
    boot_bare_spec = np.zeros(n_boot)
    boot_p2a_spec = np.zeros(n_boot)
    boot_delta = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        y = y_true[idx]
        bp = bare_preds[idx]
        pp = p2a_preds[idx]
        bs, bsp = compute_sens_spec_from_preds(bp, y)
        ps, psp = compute_sens_spec_from_preds(pp, y)
        bi = find_matched_tau_idx(bs, bsp, TARGET_SENS)
        pi = find_matched_tau_idx(ps, psp, TARGET_SENS)
        if bi < 0 or pi < 0:
            boot_bare_spec[b] = np.nan
            boot_p2a_spec[b] = np.nan
            boot_delta[b] = np.nan
        else:
            boot_bare_spec[b] = bsp[bi]
            boot_p2a_spec[b] = psp[pi]
            boot_delta[b] = psp[pi] - bsp[bi]

    def _ci(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {"mean": None, "sd": None, "ci_low": None, "ci_high": None, "n": 0}
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n": int(len(arr)),
        }

    return {
        "seed": seed,
        "n_cases": N,
        "n_bootstrap": n_boot,
        "target_sens": TARGET_SENS,
        "full_cohort": full,
        "bootstrap": {
            "bare_casespec": _ci(boot_bare_spec),
            "p2a_casespec": _ci(boot_p2a_spec),
            "delta_casespec": _ci(boot_delta),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_seeds = []
    for seed in args.seeds:
        out_path = OUT_DIR / f"bootstrap_seed{seed}.json"
        if out_path.exists() and not args.force:
            res = json.loads(out_path.read_text())
        else:
            try:
                res = bootstrap_one_seed(seed, n_boot=args.n_boot)
            except FileNotFoundError as e:
                log.error("seed=%d: %s", seed, e)
                continue
            out_path.write_text(json.dumps(res, indent=2))
            log.info("Saved: %s", out_path)
        all_seeds.append(res)

    if not all_seeds:
        log.error("No seed results")
        return

    # Aggregate across seeds
    def _across_seeds(key_path):
        vals = []
        for r in all_seeds:
            v = r
            for k in key_path:
                v = v[k]
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(v)
        arr = np.array(vals) if vals else np.array([np.nan])
        return {
            "mean_across_seeds": float(arr.mean()) if vals else None,
            "sd_across_seeds": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
            "per_seed": vals,
        }

    agg = {
        "n_seeds": len(all_seeds),
        "target_sens": TARGET_SENS,
        "n_bootstrap": args.n_boot,
        "full_cohort": {
            "bare_casespec": _across_seeds(["full_cohort", "bare_casespec"]),
            "p2a_casespec": _across_seeds(["full_cohort", "p2a_casespec"]),
            "delta_casespec": _across_seeds(["full_cohort", "delta_casespec"]),
        },
        "bootstrap_mean_per_seed": {
            "bare_casespec_mean": _across_seeds(["bootstrap", "bare_casespec", "mean"]),
            "p2a_casespec_mean": _across_seeds(["bootstrap", "p2a_casespec", "mean"]),
            "delta_casespec_mean": _across_seeds(["bootstrap", "delta_casespec", "mean"]),
            "delta_casespec_ci_low": _across_seeds(["bootstrap", "delta_casespec", "ci_low"]),
            "delta_casespec_ci_high": _across_seeds(["bootstrap", "delta_casespec", "ci_high"]),
        },
        "per_seed": all_seeds,
    }
    (OUT_DIR / "bootstrap_results.json").write_text(json.dumps(agg, indent=2))

    # Print summary
    print("\n" + "=" * 100)
    print(f"Bootstrap CI on PI-CAI fold-0 val (n=300), {args.n_boot} resamples, 5 seeds")
    print("=" * 100)
    print(f"{'seed':<6} {'Bare spec (full)':<18} {'P2a spec (full)':<18} {'ΔSpec (full)':<15} "
          f"{'ΔSpec boot mean (95%% CI)':<40}")
    for r in all_seeds:
        fc = r["full_cohort"]
        db = r["bootstrap"]["delta_casespec"]
        ci_str = f"{db['mean']:+.4f} [{db['ci_low']:+.4f}, {db['ci_high']:+.4f}]" if db["mean"] is not None else "n/a"
        print(f"{r['seed']:<6} {fc['bare_casespec']:<18.4f} {fc['p2a_casespec']:<18.4f} "
              f"{fc['delta_casespec']:<+15.4f} {ci_str:<40}")
    print("=" * 100)
    bs = agg["full_cohort"]["delta_casespec"]
    print(f"Across-seed ΔCaseSpec (full cohort): {bs['mean_across_seeds']:+.4f} ± {bs['sd_across_seeds']:.4f} "
          f"(per-seed: {bs['per_seed']})")
    print("=" * 100)


if __name__ == "__main__":
    main()
