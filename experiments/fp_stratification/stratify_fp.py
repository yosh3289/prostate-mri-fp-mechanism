#!/usr/bin/env python3
"""FP failure-mode stratification by imaging similarity to GT cancer.

For each case with >=1 FP connected component:
  1. Compute the 3-channel contrast-ratio vector (T2W, HBV, ADC) for each
     FP connected component separately (distinct from compute_fp_mimic_prostate158
     which aggregates all FPs into a single case-level triple).
  2. Compute similarity to the GT cancer contrast-ratio vector within that case
     using cosine similarity (primary; see below) and Euclidean distance.
  3. Stratify into tertiles across all FP regions pooled across 5 seeds.
  4. Report: number/proportion per tier; proportion of P2a-suppressed FPs per
     tier; proportion of residual FPs per tier.

Similarity metric choice: **cosine similarity** in 3-D contrast space.
  Rationale: T2W contrast ratios occasionally blow up (when peri-ring mean ~ 0),
  which corrupts Euclidean distance. Cosine captures *directional* agreement
  of the three-channel signature (dark T2W + high HBV + low ADC defines the
  cancer-like imaging phenotype) and is invariant to scale inflation of any
  one channel.  Cases with NaN in any channel (benign with no peri-ring, etc.)
  are excluded.

For suppression tracking:
  - A bare-A2 FP region is "suppressed" if, after P2a, no P2a FP component has
    Dice > 0.3 with the original bare FP region (i.e., the region no longer
    reaches threshold).
  - Otherwise it is "residual".

Cohorts:
  - PI-CAI fold-0 val (n=300)  -- requires bare/p2a detection sidecars from
    experiments/baselines/picai_val_{bare,p2a}_seed{S}.npz
  - Prostate158 (n=158)        -- uses existing detection sidecars in
    experiments/prostate158_p2a{,_bare}/

Outputs:
  experiments/fp_stratification/stratification_seed{S}_{cohort}.json
  experiments/fp_stratification/stratification_aggregate.json

Usage:
  conda run -n mar python experiments/fp_stratification/stratify_fp.py --cohort picai
  conda run -n mar python experiments/fp_stratification/stratify_fp.py --cohort prostate158
  conda run -n mar python experiments/fp_stratification/stratify_fp.py --cohort both
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

# Paths
MAR = Path("ADJUST_PATH/P1-MAR  (NOT in repo: detection backbone repo)")
P2B = Path("ADJUST_PATH/P2-evidence-grounded-mechanism  (NOT in repo)")
EXPERIMENTS = Path("./experiments")

PICAI_CACHE = MAR / "data/cache"
PICAI_FOLD0 = MAR / "configs/fold0_split.json"
PICAI_BARE_DET_TMPL = EXPERIMENTS / "baselines" / "picai_val_bare_seed{seed}.npz"
PICAI_P2A_DET_TMPL = EXPERIMENTS / "baselines" / "picai_val_p2a_seed{seed}.npz"

P158_CACHE = MAR / "data/prostate158_cache"
P158_BARE_DET_TMPL = EXPERIMENTS / "prostate158_p2a_bare" / "detections_seed{seed}.npz"
P158_P2A_DET_TMPL = EXPERIMENTS / "prostate158_p2a" / "detections_seed{seed}.npz"

OUT_DIR = EXPERIMENTS / "fp_stratification"

SEEDS = [42, 123, 456, 789, 1024]
DETECTION_THRESHOLD = 0.5
PERI_RING_MM = 5.0
SPACING_MM_PICAI = (3.0, 0.5, 0.5)  # (D, H, W)
SPACING_MM_P158 = (3.0, 0.5, 0.5)
MIN_CC_VOXELS = 10
DICE_SUPPRESS_THRESH = 0.3  # overlap < this => FP suppressed

sys.path.insert(0, str(P2B))
from src.data.roi_utils import get_peri_lesional_ring  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrast-ratio helpers (same formula as compute_fp_mimic_prostate158.py)
# ---------------------------------------------------------------------------
def compute_contrast_ratio(volume: np.ndarray, roi: np.ndarray, peri: np.ndarray, eps: float = 1e-6) -> float:
    if roi.sum() == 0 or peri.sum() == 0:
        return float("nan")
    m_roi = float(volume[roi.astype(bool)].mean())
    m_peri = float(volume[peri.astype(bool)].mean())
    return (m_roi - m_peri) / (abs(m_peri) + eps)


def _squeeze_3d(x) -> np.ndarray:
    arr = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# Case loaders
# ---------------------------------------------------------------------------
def load_picai_case(cache_path: Path) -> dict:
    d = torch.load(cache_path, weights_only=False)
    label = _squeeze_3d(d["label"])
    label_bool = label > 0.5
    return {
        "t2w": _squeeze_3d(d["t2w"]),
        "hbv": _squeeze_3d(d["hbv"]),
        "adc": _squeeze_3d(d["adc"]),
        "label": label_bool,
        "patient_id": d["study_id"],  # use study_id for uniqueness
        "is_positive": bool(label_bool.any()),
        "gland_mask": np.ones(label_bool.shape, dtype=bool),
    }


def load_p158_case(cache_path: Path) -> dict:
    d = torch.load(cache_path, weights_only=False)
    label = _squeeze_3d(d["label"])
    label_bool = label > 0.5
    return {
        "t2w": _squeeze_3d(d["t2w"]),
        "hbv": _squeeze_3d(d["hbv"]),
        "adc": _squeeze_3d(d["adc"]),
        "label": label_bool,
        "patient_id": d["patient_id"],
        "is_positive": bool(d.get("is_positive", label_bool.any())),
        "gland_mask": np.ones(label_bool.shape, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Per-FP-component analysis
# ---------------------------------------------------------------------------
def get_fp_components(detection: np.ndarray, gt: np.ndarray,
                      threshold: float = DETECTION_THRESHOLD,
                      min_voxels: int = MIN_CC_VOXELS) -> list[np.ndarray]:
    """Return list of boolean masks, one per FP connected component (size >= min_voxels)."""
    fp_raw = (detection > threshold) & (~gt)
    if fp_raw.sum() == 0:
        return []
    labeled, n = ndimage.label(fp_raw)
    if n == 0:
        return []
    comps = []
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_voxels:
            comps.append(comp)
    return comps


def compute_region_contrast_triple(
    case: dict,
    region_mask: np.ndarray,
    spacing_mm: tuple,
) -> tuple[float, float, float]:
    """Return (t2w_cr, hbv_cr, adc_cr) for a single region."""
    if not region_mask.any():
        return (float("nan"), float("nan"), float("nan"))
    peri = get_peri_lesional_ring(
        region_mask.astype(np.uint8), case["gland_mask"].astype(np.uint8),
        dilation_mm=PERI_RING_MM, spacing_mm=spacing_mm,
    ).astype(bool)
    t2w = compute_contrast_ratio(case["t2w"], region_mask, peri)
    hbv = compute_contrast_ratio(case["hbv"], region_mask, peri)
    adc = compute_contrast_ratio(case["adc"], region_mask, peri)
    return (t2w, hbv, adc)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


# ---------------------------------------------------------------------------
# FP overlap (suppression check)
# ---------------------------------------------------------------------------
def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return 2.0 * np.logical_and(a, b).sum() / denom


def is_suppressed(bare_fp_mask: np.ndarray, p2a_fp_components: list[np.ndarray]) -> bool:
    """A bare FP region is 'suppressed' if no P2a FP component has Dice > 0.3 with it."""
    for p2a_comp in p2a_fp_components:
        if dice_score(bare_fp_mask, p2a_comp) > DICE_SUPPRESS_THRESH:
            return False
    return True


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------
def run_seed(seed: int, cohort: str) -> dict:
    """Return dict with per_fp records (one per bare-FP CC) plus metadata.

    Each record:
        {case_id, t2w_cr, hbv_cr, adc_cr, gt_t2w_cr, gt_hbv_cr, gt_adc_cr,
         cosine_sim, euclid_dist, is_suppressed, fp_voxels}
    """
    if cohort == "picai":
        bare_path = Path(str(PICAI_BARE_DET_TMPL).format(seed=seed))
        p2a_path = Path(str(PICAI_P2A_DET_TMPL).format(seed=seed))
        cache_dir = PICAI_CACHE
        spacing = SPACING_MM_PICAI
        with PICAI_FOLD0.open() as f:
            split = json.load(f)
        case_ids = split["val"]
        load_case = load_picai_case
    elif cohort == "prostate158":
        bare_path = Path(str(P158_BARE_DET_TMPL).format(seed=seed))
        p2a_path = Path(str(P158_P2A_DET_TMPL).format(seed=seed))
        cache_dir = P158_CACHE
        spacing = SPACING_MM_P158
        case_ids = [f.stem for f in sorted(cache_dir.glob("*.pt"))]
        load_case = load_p158_case
    else:
        raise ValueError(f"Unknown cohort: {cohort}")

    if not bare_path.exists():
        raise FileNotFoundError(f"Bare detection sidecar missing: {bare_path}")
    if not p2a_path.exists():
        raise FileNotFoundError(f"P2a detection sidecar missing: {p2a_path}")

    log.info("Loading detections (seed=%d, cohort=%s)", seed, cohort)
    bare_det = dict(np.load(bare_path))
    p2a_det = dict(np.load(p2a_path))

    records = []
    n_cases_with_fp = 0
    n_cases = 0

    for cid in case_ids:
        cache_path = cache_dir / f"{cid}.pt"
        if not cache_path.exists():
            log.warning("Missing cache file: %s", cache_path)
            continue
        n_cases += 1
        case = load_case(cache_path)
        case_key = case["patient_id"]
        # The npz keys are study_id for PICAI and patient_id for P158 --
        # but for PICAI we set patient_id to study_id already.
        # The P158 bare/p2a sidecars use patient_id.  Use case_key.
        key_candidates = [case_key, cid]
        det_key = None
        for k in key_candidates:
            if k in bare_det.files if hasattr(bare_det, "files") else k in bare_det:
                det_key = k
                break
        if det_key is None:
            # dict(np.load(...)) returns plain dict
            for k in key_candidates:
                if k in bare_det:
                    det_key = k
                    break
        if det_key is None:
            log.warning("No bare detection for %s (tried %s)", case_key, key_candidates)
            continue

        if det_key not in p2a_det:
            log.warning("No p2a detection for %s", det_key)
            continue

        bare_map = np.squeeze(bare_det[det_key])
        p2a_map = np.squeeze(p2a_det[det_key])

        if bare_map.shape != case["label"].shape or p2a_map.shape != case["label"].shape:
            log.warning("Shape mismatch for %s bare=%s p2a=%s label=%s",
                        case_key, bare_map.shape, p2a_map.shape, case["label"].shape)
            continue

        # GT contrast vector  (needed for similarity)
        gt = case["label"]
        if not gt.any():
            gt_cr = None  # negative case: no GT reference; we'll skip similarity
        else:
            gt_cr = np.array(compute_region_contrast_triple(case, gt, spacing))
            if np.any(np.isnan(gt_cr)):
                gt_cr = None

        # FP components for bare and p2a
        bare_fps = get_fp_components(bare_map, gt)
        p2a_fps = get_fp_components(p2a_map, gt)

        if bare_fps:
            n_cases_with_fp += 1

        for fp_mask in bare_fps:
            t2w_cr, hbv_cr, adc_cr = compute_region_contrast_triple(case, fp_mask, spacing)
            fp_cr = np.array([t2w_cr, hbv_cr, adc_cr])
            # Similarity only defined if BOTH GT and FP CR are finite
            if gt_cr is not None and np.all(np.isfinite(fp_cr)):
                cos = cosine_similarity(fp_cr, gt_cr)
                euc = euclidean_distance(fp_cr, gt_cr)
            else:
                cos = float("nan")
                euc = float("nan")

            suppressed = is_suppressed(fp_mask, p2a_fps)

            records.append({
                "case_id": case_key,
                "t2w_cr": t2w_cr,
                "hbv_cr": hbv_cr,
                "adc_cr": adc_cr,
                "gt_t2w_cr": float(gt_cr[0]) if gt_cr is not None else float("nan"),
                "gt_hbv_cr": float(gt_cr[1]) if gt_cr is not None else float("nan"),
                "gt_adc_cr": float(gt_cr[2]) if gt_cr is not None else float("nan"),
                "cosine_sim_to_gt": cos,
                "euclidean_dist_to_gt": euc,
                "is_suppressed": bool(suppressed),
                "fp_voxels": int(fp_mask.sum()),
                "case_has_gt": bool(gt.any()),
            })

    log.info("seed=%d cohort=%s: n_cases=%d n_cases_with_fp=%d n_fp_regions=%d",
             seed, cohort, n_cases, n_cases_with_fp, len(records))

    return {
        "seed": seed,
        "cohort": cohort,
        "n_cases": n_cases,
        "n_cases_with_fp": n_cases_with_fp,
        "n_fp_regions": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Aggregation + stratification
# ---------------------------------------------------------------------------
def stratify_and_summarize(seed_results: list[dict], cohort: str) -> dict:
    """Pool all FPs across 5 seeds, compute tertiles of cosine similarity,
    assign tiers, report counts / suppression rates per tier per seed."""
    all_recs = []
    for r in seed_results:
        for rec in r["records"]:
            rec2 = dict(rec)
            rec2["seed"] = r["seed"]
            all_recs.append(rec2)

    # Similarity distribution (cases where cosine_sim is finite)
    sims = np.array([r["cosine_sim_to_gt"] for r in all_recs], dtype=np.float64)
    finite = np.isfinite(sims)
    n_finite = int(finite.sum())
    n_nan = int((~finite).sum())

    if n_finite < 3:
        log.error("Only %d finite similarity values -- cannot compute tertiles", n_finite)
        return {
            "cohort": cohort,
            "error": "insufficient_finite_similarities",
            "n_fp_total": len(all_recs),
            "n_finite": n_finite,
        }

    p33, p67 = np.percentile(sims[finite], [33.333, 66.667])
    log.info("[%s] tertile boundaries (cosine sim): p33=%.4f p67=%.4f", cohort, p33, p67)

    def assign_tier(s: float) -> str:
        if not np.isfinite(s):
            return "nan"
        if s <= p33:
            return "low"
        elif s <= p67:
            return "mid"
        else:
            return "high"

    # Tag every record
    for r in all_recs:
        r["tier"] = assign_tier(r["cosine_sim_to_gt"])

    # Summary aggregates
    tiers = ["low", "mid", "high", "nan"]

    def _count(recs, tier=None, suppressed=None):
        c = 0
        for r in recs:
            if tier is not None and r["tier"] != tier:
                continue
            if suppressed is not None and r["is_suppressed"] != suppressed:
                continue
            c += 1
        return c

    # Per-seed tier breakdown
    per_seed_summary = {}
    for sr in seed_results:
        seed = sr["seed"]
        seed_recs = [r for r in all_recs if r["seed"] == seed]
        tier_stats = {}
        for t in tiers:
            n_total = _count(seed_recs, tier=t)
            n_suppressed = _count(seed_recs, tier=t, suppressed=True)
            n_residual = _count(seed_recs, tier=t, suppressed=False)
            tier_stats[t] = {
                "n": n_total,
                "n_suppressed": n_suppressed,
                "n_residual": n_residual,
                "suppression_rate": (n_suppressed / n_total) if n_total > 0 else float("nan"),
            }
        # Among suppressed FPs, what fraction came from each tier?
        total_suppressed = _count(seed_recs, suppressed=True)
        total_residual = _count(seed_recs, suppressed=False)
        for t in tiers:
            ns = tier_stats[t]["n_suppressed"]
            nr = tier_stats[t]["n_residual"]
            tier_stats[t]["share_of_suppressed"] = (ns / total_suppressed) if total_suppressed > 0 else float("nan")
            tier_stats[t]["share_of_residual"] = (nr / total_residual) if total_residual > 0 else float("nan")
        per_seed_summary[f"seed{seed}"] = {
            "n_fp_total": len(seed_recs),
            "n_suppressed_total": total_suppressed,
            "n_residual_total": total_residual,
            "suppression_rate_overall": (total_suppressed / len(seed_recs)) if seed_recs else float("nan"),
            "tiers": tier_stats,
        }

    # Pooled aggregate
    pooled_tier_stats = {}
    for t in tiers:
        n_total = _count(all_recs, tier=t)
        n_suppressed = _count(all_recs, tier=t, suppressed=True)
        n_residual = _count(all_recs, tier=t, suppressed=False)
        pooled_tier_stats[t] = {
            "n": n_total,
            "n_suppressed": n_suppressed,
            "n_residual": n_residual,
            "suppression_rate": (n_suppressed / n_total) if n_total > 0 else float("nan"),
            "proportion_of_all_fps": (n_total / len(all_recs)) if all_recs else float("nan"),
        }
    total_suppressed_pooled = _count(all_recs, suppressed=True)
    total_residual_pooled = _count(all_recs, suppressed=False)
    for t in tiers:
        ns = pooled_tier_stats[t]["n_suppressed"]
        nr = pooled_tier_stats[t]["n_residual"]
        pooled_tier_stats[t]["share_of_suppressed"] = (ns / total_suppressed_pooled) if total_suppressed_pooled > 0 else float("nan")
        pooled_tier_stats[t]["share_of_residual"] = (nr / total_residual_pooled) if total_residual_pooled > 0 else float("nan")

    return {
        "cohort": cohort,
        "n_fp_total": len(all_recs),
        "n_finite_similarity": n_finite,
        "n_nan_similarity": n_nan,
        "tertile_boundaries_cosine": {"p33": float(p33), "p67": float(p67)},
        "pooled": {
            "n_suppressed_total": total_suppressed_pooled,
            "n_residual_total": total_residual_pooled,
            "suppression_rate_overall": (total_suppressed_pooled / len(all_recs)) if all_recs else float("nan"),
            "tiers": pooled_tier_stats,
        },
        "per_seed": per_seed_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["picai", "prostate158", "both"], default="both")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = ["picai", "prostate158"] if args.cohort == "both" else [args.cohort]
    aggregate = {}

    for cohort in cohorts:
        log.info("===== COHORT: %s =====", cohort)
        seed_results = []
        for seed in args.seeds:
            out_path = OUT_DIR / f"stratification_seed{seed}_{cohort}.json"
            if out_path.exists() and not args.force:
                log.info("Loading cached: %s", out_path)
                with out_path.open() as f:
                    sr = json.load(f)
            else:
                try:
                    sr = run_seed(seed, cohort)
                except FileNotFoundError as e:
                    log.error("Seed %d %s: %s", seed, cohort, e)
                    continue
                with out_path.open("w") as f:
                    json.dump(sr, f, indent=2, default=float)
                log.info("Saved: %s", out_path)
            seed_results.append(sr)

        if not seed_results:
            log.error("No seed results collected for cohort=%s", cohort)
            continue

        agg = stratify_and_summarize(seed_results, cohort)
        aggregate[cohort] = agg

        # Pretty summary
        log.info("===== Stratification summary (%s, pooled) =====", cohort)
        log.info("Total FPs: %d | finite similarity: %d | nan: %d",
                 agg["n_fp_total"], agg["n_finite_similarity"], agg["n_nan_similarity"])
        for t in ["high", "mid", "low", "nan"]:
            s = agg["pooled"]["tiers"][t]
            log.info("  %-5s (n=%4d, %.1f%% of all): suppression_rate=%.3f  share_of_suppressed=%.3f  share_of_residual=%.3f",
                     t, s["n"], 100 * s.get("proportion_of_all_fps", 0), s["suppression_rate"],
                     s.get("share_of_suppressed", float("nan")), s.get("share_of_residual", float("nan")))

    agg_path = OUT_DIR / "stratification_aggregate.json"
    with agg_path.open("w") as f:
        json.dump(aggregate, f, indent=2, default=float)
    log.info("Aggregate written: %s", agg_path)

    # Printed summary table
    print("\n" + "=" * 100)
    print("FP Failure-Mode Stratification Summary (pooled across 5 seeds)")
    print("=" * 100)
    for cohort, agg in aggregate.items():
        print(f"\nCohort: {cohort}")
        print(f"  Total FPs: {agg['n_fp_total']} (finite sim: {agg['n_finite_similarity']}, nan: {agg['n_nan_similarity']})")
        print(f"  Tertile boundaries (cosine sim): p33={agg['tertile_boundaries_cosine']['p33']:.4f}  p67={agg['tertile_boundaries_cosine']['p67']:.4f}")
        print(f"  {'tier':<6} {'n':>6} {'%-all':>7} {'supp%':>7} {'share-supp':>11} {'share-res':>10}")
        for t in ["high", "mid", "low", "nan"]:
            s = agg["pooled"]["tiers"][t]
            pa = 100 * s.get("proportion_of_all_fps", 0)
            sr = 100 * s["suppression_rate"] if np.isfinite(s["suppression_rate"]) else float("nan")
            ss = 100 * s.get("share_of_suppressed", 0) if np.isfinite(s.get("share_of_suppressed", float("nan"))) else float("nan")
            srs = 100 * s.get("share_of_residual", 0) if np.isfinite(s.get("share_of_residual", float("nan"))) else float("nan")
            def _fmt(v):
                return f"{v:7.2f}" if np.isfinite(v) else "    nan"
            print(f"  {t:<6} {s['n']:>6} {_fmt(pa)} {_fmt(sr)} {_fmt(ss):>11} {_fmt(srs):>10}")
    print("=" * 100)


if __name__ == "__main__":
    main()
