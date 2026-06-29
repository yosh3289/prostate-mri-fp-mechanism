#!/usr/bin/env python3
"""Relative specificity across discrete sensitivity targets on Prostate158.

Efficient single-pass design:
  1. For each (model, seed), sweep all thresholds ONCE, caching (tau, sens, spec).
  2. For each target_sens in {0.80, 0.85, 0.90, 0.95}, pick the cached row with
     sens closest to the target (ties broken by higher spec).
  3. Record reached / unreached_max / saturated_min status.

Aggregates per-target mean +/- sd across seeds.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHED_SENS_DIR = REPO_ROOT / "experiments" / "prostate158_matched_sens"
sys.path.insert(0, str(MATCHED_SENS_DIR))

from compute_matched_sens import compute_casesens_casespec, load_gt_labels  # type: ignore

P2A_DIR = REPO_ROOT / "experiments" / "prostate158_p2a"
BARE_DIR = REPO_ROOT / "experiments" / "prostate158_p2a_bare"
OUT_DIR = REPO_ROOT / "experiments" / "prostate158_sensspec_curve"

SEEDS = [42, 123, 456, 789, 1024]
TARGETS = [0.80, 0.85, 0.90, 0.95]
TOLERANCE = 0.02

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def sweep_all_thresholds(detections, gt, thresholds=None):
    """Sweep all thresholds ONCE; return list of (tau, sens, spec) triples, ascending by tau."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    curve = []
    for t in thresholds:
        sens, spec, _, _ = compute_casesens_casespec(detections, gt, float(t))
        if np.isnan(sens):
            continue
        curve.append((float(t), float(sens), float(spec)))
    return curve


def pick_target(curve, target_sens, tolerance=TOLERANCE):
    """From cached curve, pick the operating point with sens closest to target.
    Returns (tau, achieved_sens, achieved_spec, status).
    """
    best = None
    best_diff = float("inf")
    for tau, sens, spec in curve:
        diff = abs(sens - target_sens)
        if diff < best_diff or (diff == best_diff and best is not None and spec > best[2]):
            best = (tau, sens, spec)
            best_diff = diff
    if best is None:
        raise RuntimeError("Empty curve")
    tau, sens, spec = best
    if sens >= target_sens - tolerance and sens <= target_sens + tolerance:
        status = "reached"
    elif sens < target_sens - tolerance:
        status = "unreached_max"
    else:
        status = "saturated_min"
    return tau, sens, spec, status


def process_model(name: str, det_dir: Path, gt: dict, seeds: list[int]) -> dict:
    per_seed_points = []
    summary = {str(t): {"specs": [], "senses": [], "statuses": []} for t in TARGETS}
    for seed in seeds:
        det_path = det_dir / f"detections_seed{seed}.npz"
        if not det_path.exists():
            log.error("%s seed %d detection sidecar missing: %s", name, seed, det_path)
            continue
        detections = {k: v for k, v in np.load(det_path).items()}
        log.info("%s seed %d: sweeping 99 thresholds x %d cases ...", name, seed, len(detections))
        curve = sweep_all_thresholds(detections, gt)
        sens_min = min(s for _, s, _ in curve)
        sens_max = max(s for _, s, _ in curve)
        log.info("   curve sens range: %.3f -> %.3f (n=%d pts)", sens_min, sens_max, len(curve))

        seed_record = {"seed": seed, "curve": [
            {"tau": t, "sens": s, "spec": sp} for t, s, sp in curve
        ], "picks": {}}
        for target in TARGETS:
            tau, sens, spec, status = pick_target(curve, target)
            seed_record["picks"][str(target)] = {
                "tau": tau, "achieved_sens": sens, "achieved_spec": spec, "status": status,
            }
            summary[str(target)]["specs"].append(spec)
            summary[str(target)]["senses"].append(sens)
            summary[str(target)]["statuses"].append(status)
        per_seed_points.append(seed_record)

    agg = {}
    for target in TARGETS:
        key = str(target)
        specs_all = summary[key]["specs"]
        senses_all = summary[key]["senses"]
        statuses = summary[key]["statuses"]
        specs_reached = [
            s for s, st in zip(specs_all, statuses) if st == "reached"
        ]
        agg[key] = {
            "mean_casespec_reached": float(np.mean(specs_reached)) if specs_reached else None,
            "sd_casespec_reached": float(np.std(specs_reached, ddof=1)) if len(specs_reached) > 1 else None,
            "n_seeds_reached": len(specs_reached),
            "mean_casespec_all": float(np.mean(specs_all)) if specs_all else None,
            "sd_casespec_all": float(np.std(specs_all, ddof=1)) if len(specs_all) > 1 else None,
            "per_seed_sens": senses_all,
            "per_seed_spec": specs_all,
            "per_seed_status": statuses,
        }

    return {"per_seed": per_seed_points, "summary": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading GT labels ...")
    gt = load_gt_labels()
    log.info("Loaded %d cases (%d positive)", len(gt), sum(1 for _, pos in gt.values() if pos))

    log.info("=== Bare A2 ===")
    bare_out = process_model("bare", BARE_DIR, gt, args.seeds)
    log.info("=== P2a ms v2 ===")
    p2a_out = process_model("p2a", P2A_DIR, gt, args.seeds)

    # Per-seed JSONs
    for seed in args.seeds:
        bare_seed = next((r for r in bare_out["per_seed"] if r["seed"] == seed), None)
        p2a_seed = next((r for r in p2a_out["per_seed"] if r["seed"] == seed), None)
        with (OUT_DIR / f"sensspec_seed{seed}.json").open("w") as f:
            json.dump({"seed": seed, "bare": bare_seed, "p2a": p2a_seed}, f, indent=2)

    # Aggregate
    agg = {
        "targets": TARGETS,
        "tolerance": TOLERANCE,
        "n_seeds_total": len(args.seeds),
        "bare": bare_out["summary"],
        "p2a": p2a_out["summary"],
    }
    with (OUT_DIR / "sensspec_aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)
    log.info("Wrote aggregate")

    # Summary table
    print("\n=== Summary Table (reached targets only) ===")
    print(f"{'target':>8} {'bare_spec':>18} {'bare_n_ok':>10} {'p2a_spec':>18} {'p2a_n_ok':>10}")
    for t in TARGETS:
        b = bare_out["summary"][str(t)]
        p = p2a_out["summary"][str(t)]
        def _fmt(s):
            if s["mean_casespec_reached"] is None:
                return "n/a"
            sd = s["sd_casespec_reached"] if s["sd_casespec_reached"] is not None else 0.0
            return f"{s['mean_casespec_reached']:.3f}+/-{sd:.3f}"
        print(f"{t:>8.2f} {_fmt(b):>18} {b['n_seeds_reached']:>10} {_fmt(p):>18} {p['n_seeds_reached']:>10}")


if __name__ == "__main__":
    main()
