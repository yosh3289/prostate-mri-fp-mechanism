#!/usr/bin/env python3
"""Matched-sensitivity threshold sweep on Prostate158.

For each seed, finds the Prostate158 threshold that reproduces each model's
PI-CAI validation CaseSens (at the standard detection threshold), then
reports CaseSpec at that threshold.

A paired Wilcoxon signed-rank test (one-sided, P2a > bare) tests whether
the specificity improvement is significant across the 5 random seeds.

PI-CAI CaseSens values (looked up from PI-CAI ideal-scenario eval JSONs,
NOT defaulted) -- used as the per-seed matching target:
  Bare A2:
    seed  42: 0.9405   (P1-MAR outputs/results/ablation_nnunet/A2/seed42/adaptive_nnunet_eval.json)
    seed 123: 0.9405
    seed 456: 0.9405
    seed 789: 0.9762
    seed1024: 0.9167
  P2a (fp_suppression_ms_v2):
    seed  42: 0.9405   (P2 outputs/eval_results/fp_suppression_ms_v2_seed*.json)
    seed 123: 0.9405
    seed 456: 0.9405
    seed 789: 0.9762
    seed1024: 0.9167

Both models share identical per-seed PI-CAI CaseSens values (same operating
point from the calibrated threshold), so the matching target is the same for
both models within each seed.

Inputs:
  experiments/prostate158_p2a_bare/detections_seed{S}.npz  (bare A2)
  experiments/prostate158_p2a/detections_seed{S}.npz       (P2a v2 ms)
  Prostate158 cache for ground-truth labels

Outputs:
  experiments/prostate158_matched_sens/matched_sens_seed{S}.json
  experiments/prostate158_matched_sens/matched_sens_aggregate.json

Usage:
  conda run -n mar python experiments/prostate158_matched_sens/compute_matched_sens.py
  conda run -n mar python experiments/prostate158_matched_sens/compute_matched_sens.py \\
      --target-sens-bare 0.94 --target-sens-p2a 0.94   # override with common value
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
P2A_DIR  = EXPERIMENTS_DIR / "prostate158_p2a"
BARE_DIR = EXPERIMENTS_DIR / "prostate158_p2a_bare"
P158_CACHE = Path(
    "ADJUST_PATH/prostate158_cache  (NOT in repo: local Prostate158 preprocessed cache)"
)
OUT_DIR = Path(__file__).resolve().parent

SEEDS = [42, 123, 456, 789, 1024]

# Per-seed PI-CAI CaseSens (looked up from actual eval JSONs -- NOT defaulted).
# Both models share the same operating point per seed (same threshold was
# chosen for both during the PI-CAI validation campaign).
PICAI_CASESENS_BARE = {
    42:   0.9404761904761905,
    123:  0.9404761904761905,
    456:  0.9404761904761905,
    789:  0.9761904761904762,
    1024: 0.9166666666666666,
}
PICAI_CASESENS_P2A = {
    42:   0.9404761904761905,
    123:  0.9404761904761905,
    456:  0.9404761904761905,
    789:  0.9761904761904762,
    1024: 0.9166666666666666,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------
def load_gt_labels() -> dict[str, tuple[np.ndarray, bool]]:
    """Returns {patient_id: (label_3d_bool_array, is_positive)}."""
    out: dict[str, tuple[np.ndarray, bool]] = {}
    for f in sorted(P158_CACHE.glob("*.pt")):
        d = torch.load(f, weights_only=False)
        pid = d["patient_id"]
        label = d["label"]
        arr = label.numpy() if hasattr(label, "numpy") else np.asarray(label)
        while arr.ndim > 3:
            arr = arr[0]
        arr_bool = arr > 0.5
        out[pid] = (arr_bool, bool(arr_bool.any()))
    return out


# ---------------------------------------------------------------------------
# Case-level prediction
# ---------------------------------------------------------------------------
def compute_casesens_casespec(
    detections: dict[str, np.ndarray],
    gt: dict[str, tuple[np.ndarray, bool]],
    threshold: float,
    min_voxels: int = 10,
) -> tuple[float, float, int, int]:
    """Compute CaseSens and CaseSpec at a given threshold over all cases.

    Returns (CaseSens, CaseSpec, n_positive, n_negative).
    A case is predicted positive if any connected component of
    (detection > threshold) has >= min_voxels voxels.
    """
    tp = fp = tn = fn = 0
    for pid, det in detections.items():
        if pid not in gt:
            log.warning("Patient %s not in GT -- skipped", pid)
            continue
        label_arr, is_pos = gt[pid]
        binary = det > threshold
        if not binary.any():
            pred_pos = False
        else:
            labeled, n = ndimage.label(binary)
            if n == 0:
                pred_pos = False
            else:
                sizes = ndimage.sum(binary, labeled, range(1, n + 1))
                pred_pos = any(s >= min_voxels for s in sizes)

        if is_pos and pred_pos:
            tp += 1
        elif is_pos and not pred_pos:
            fn += 1
        elif not is_pos and pred_pos:
            fp += 1
        else:
            tn += 1

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec, tp + fn, tn + fp


# ---------------------------------------------------------------------------
# Threshold matching
# ---------------------------------------------------------------------------
def find_threshold_matching_sens(
    detections: dict[str, np.ndarray],
    gt: dict[str, tuple[np.ndarray, bool]],
    target_sens: float,
    min_voxels: int = 10,
) -> tuple[float, float, float]:
    """Grid-search for threshold that minimises |CaseSens - target_sens|.

    Searches tau in [0.01, 0.99] with step 0.01.
    Ties in |sens - target| are broken by higher CaseSpec (more specific).

    Returns (best_tau, achieved_CaseSens, achieved_CaseSpec).
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_tau: float | None = None
    best_sens: float = float("nan")
    best_spec: float = float("nan")
    best_diff: float = float("inf")

    for t in thresholds:
        sens, spec, _, _ = compute_casesens_casespec(detections, gt, float(t), min_voxels)
        if np.isnan(sens):
            continue
        diff = abs(sens - target_sens)
        # Prefer smaller diff; tie-break by higher spec
        if diff < best_diff - 1e-9 or (
            abs(diff - best_diff) < 1e-9 and best_spec is not float("nan") and spec > best_spec
        ):
            best_tau, best_sens, best_spec = float(t), sens, spec
            best_diff = diff

    if best_tau is None:
        raise RuntimeError("No valid threshold found for target_sens=%f" % target_sens)
    return best_tau, best_sens, best_spec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute matched-sensitivity CaseSpec comparison (P2a vs bare A2) on Prostate158."
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument(
        "--target-sens-bare", type=float, default=None,
        help=(
            "Override PI-CAI CaseSens target for bare A2 (same value applied to all seeds). "
            "If omitted, per-seed values looked up from PI-CAI eval JSONs are used."
        ),
    )
    ap.add_argument(
        "--target-sens-p2a", type=float, default=None,
        help="Override PI-CAI CaseSens target for P2a (same value applied to all seeds).",
    )
    ap.add_argument("--min-voxels", type=int, default=10,
                    help="Minimum component voxels to count as a detection (default 10).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine per-seed targets
    use_override_bare = args.target_sens_bare is not None
    use_override_p2a  = args.target_sens_p2a  is not None
    if use_override_bare:
        log.info("Using OVERRIDE target CaseSens for bare A2: %.4f (all seeds)", args.target_sens_bare)
    else:
        log.info("Using LOOKED-UP per-seed PI-CAI CaseSens targets for bare A2: %s",
                 PICAI_CASESENS_BARE)
    if use_override_p2a:
        log.info("Using OVERRIDE target CaseSens for P2a: %.4f (all seeds)", args.target_sens_p2a)
    else:
        log.info("Using LOOKED-UP per-seed PI-CAI CaseSens targets for P2a: %s",
                 PICAI_CASESENS_P2A)

    log.info("Loading Prostate158 ground-truth labels from %s ...", P158_CACHE)
    gt = load_gt_labels()
    n_pos = sum(1 for _, pos in gt.values() if pos)
    log.info("Loaded GT for %d cases (%d positive, %d negative)", len(gt), n_pos, len(gt) - n_pos)

    per_seed_results: list[dict] = []

    for seed in args.seeds:
        bare_det_path = BARE_DIR / f"detections_seed{seed}.npz"
        p2a_det_path  = P2A_DIR  / f"detections_seed{seed}.npz"

        if not bare_det_path.exists():
            log.error(
                "Bare detection sidecar missing for seed %d: %s\n"
                "  Run: conda run -n mar python experiments/prostate158_p2a_bare/eval_prostate158_bare.py --seeds %d",
                seed, bare_det_path, seed,
            )
            continue
        if not p2a_det_path.exists():
            log.error("P2a detection sidecar missing for seed %d: %s", seed, p2a_det_path)
            continue

        log.info("Loading detections for seed %d ...", seed)
        bare_det = dict(np.load(bare_det_path))
        p2a_det  = dict(np.load(p2a_det_path))
        log.info("  bare: %d cases  |  p2a: %d cases", len(bare_det), len(p2a_det))

        # Determine targets for this seed
        target_bare = args.target_sens_bare if use_override_bare else PICAI_CASESENS_BARE.get(seed, 0.94)
        target_p2a  = args.target_sens_p2a  if use_override_p2a  else PICAI_CASESENS_P2A.get(seed, 0.94)

        log.info("=== Seed %d  (target_bare=%.4f  target_p2a=%.4f) ===", seed, target_bare, target_p2a)

        tau_bare, sens_bare, spec_bare = find_threshold_matching_sens(
            bare_det, gt, target_bare, args.min_voxels
        )
        tau_p2a, sens_p2a, spec_p2a = find_threshold_matching_sens(
            p2a_det, gt, target_p2a, args.min_voxels
        )
        delta = spec_p2a - spec_bare

        log.info(
            "  bare : tau=%.2f  CaseSens=%.4f (target %.4f)  CaseSpec=%.4f",
            tau_bare, sens_bare, target_bare, spec_bare,
        )
        log.info(
            "  p2a  : tau=%.2f  CaseSens=%.4f (target %.4f)  CaseSpec=%.4f",
            tau_p2a, sens_p2a, target_p2a, spec_p2a,
        )
        log.info("  delta CaseSpec (p2a - bare) = %+.4f", delta)

        seed_result = {
            "seed": seed,
            "target_casesens_bare": target_bare,
            "target_casesens_p2a": target_p2a,
            "target_source": "looked_up" if (not use_override_bare and not use_override_p2a) else "override",
            "tau_match_bare": tau_bare,
            "tau_match_p2a": tau_p2a,
            "achieved_casesens_bare": sens_bare,
            "achieved_casesens_p2a": sens_p2a,
            "casespec_bare_matched": spec_bare,
            "casespec_p2a_matched": spec_p2a,
            "delta_casespec": delta,
            "min_voxels": args.min_voxels,
        }
        out_path = OUT_DIR / f"matched_sens_seed{seed}.json"
        with out_path.open("w") as f:
            json.dump(seed_result, f, indent=2)
        log.info("  Written: %s", out_path)
        per_seed_results.append(seed_result)

    # ------------------------------------------------------------------
    # Aggregate + paired Wilcoxon signed-rank test (one-sided: p2a > bare)
    # ------------------------------------------------------------------
    if len(per_seed_results) < 2:
        log.error("Only %d seeds completed -- cannot compute aggregate or Wilcoxon test.", len(per_seed_results))
        return

    spec_bare_vals = np.array([s["casespec_bare_matched"] for s in per_seed_results])
    spec_p2a_vals  = np.array([s["casespec_p2a_matched"]  for s in per_seed_results])
    deltas         = np.array([s["delta_casespec"]         for s in per_seed_results])
    seeds_done     = [s["seed"]                            for s in per_seed_results]

    try:
        w_stat, w_p = stats.wilcoxon(
            spec_p2a_vals, spec_bare_vals,
            alternative="greater",
            zero_method="zsplit",
        )
        w_stat = float(w_stat)
        w_p    = float(w_p)
    except ValueError as e:
        log.warning("Wilcoxon test failed (%s) -- likely all differences are zero.", e)
        w_stat, w_p = float("nan"), float("nan")

    agg = {
        "n_seeds": len(per_seed_results),
        "seeds": seeds_done,
        "min_voxels": args.min_voxels,
        "target_source": per_seed_results[0].get("target_source", "looked_up"),
        "casespec_bare_matched": {
            "mean":     float(spec_bare_vals.mean()),
            "sd":       float(spec_bare_vals.std(ddof=1)),
            "per_seed": spec_bare_vals.tolist(),
        },
        "casespec_p2a_matched": {
            "mean":     float(spec_p2a_vals.mean()),
            "sd":       float(spec_p2a_vals.std(ddof=1)),
            "per_seed": spec_p2a_vals.tolist(),
        },
        "delta_casespec": {
            "mean":     float(deltas.mean()),
            "sd":       float(deltas.std(ddof=1)),
            "per_seed": deltas.tolist(),
        },
        "wilcoxon_one_sided_greater": {
            "description": "Paired Wilcoxon signed-rank test: H1 = P2a CaseSpec > bare CaseSpec at matched sensitivity",
            "statistic": w_stat if not np.isnan(w_stat) else None,
            "p_value":   w_p    if not np.isnan(w_p)    else None,
        },
    }

    agg_path = OUT_DIR / "matched_sens_aggregate.json"
    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2)
    log.info("Written aggregate: %s", agg_path)

    log.info("=== AGGREGATE MATCHED-SENSITIVITY RESULTS ===")
    log.info("  CaseSpec bare  (matched): %.4f ± %.4f", agg["casespec_bare_matched"]["mean"], agg["casespec_bare_matched"]["sd"])
    log.info("  CaseSpec P2a   (matched): %.4f ± %.4f", agg["casespec_p2a_matched"]["mean"],  agg["casespec_p2a_matched"]["sd"])
    log.info("  ΔCaseSpec (P2a - bare):   %+.4f ± %.4f", agg["delta_casespec"]["mean"],      agg["delta_casespec"]["sd"])
    log.info("  Wilcoxon p (P2a > bare):  %s",
             f"{w_p:.4f}" if not np.isnan(w_p) else "nan (test failed)")


if __name__ == "__main__":
    main()
