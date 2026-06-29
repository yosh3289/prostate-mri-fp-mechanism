#!/usr/bin/env python3
"""Replicate P2b FP-mimic contrast analysis on Prostate158.

Computes T2W, ADC, HBV contrast ratios (ROI vs 5mm peri-lesional ring) for:
  - GT lesion ROIs (positive cases only)
  - P1 FP ROIs (detection regions not overlapping GT)
  - Contralateral benign ROIs (mirror of GT, positive cases only)

Contrast ratio formula:
    (mean_ROI - mean_peri) / (|mean_peri| + eps)

This is the simpler Prostate158-adapted version of PI-CAI validate_fp_evidence.py.
No pre-computed gland/instance/benign masks are needed; all ROIs are derived on the
fly from cache contents. Since Prostate158 cache has no gland_mask, a full-volume
surrogate (all-ones) is used, consistent with the task specification.

For each seed, outputs:
    experiments/prostate158_p2b/fp_mimic_seed{SEED}.json
with:
  - per_case: list of {id, has_gt, fp_voxels, benign_voxels,
                       t2w_gt/fp/benign, adc_gt/fp/benign, hbv_gt/fp/benign}
  - summary: per-ROI-type mean±SD across cases for each modality
  - seed, n_cases

Usage:
    conda run -n mar python experiments/prostate158_p2b/compute_fp_mimic_prostate158.py --seeds 42
    conda run -n mar python experiments/prostate158_p2b/compute_fp_mimic_prostate158.py  # all 5 seeds
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────────────
MAR = Path("ADJUST_PATH/P1-MAR  (NOT in repo: detection backbone repo)")
P2B = Path("ADJUST_PATH/P2-evidence-grounded-mechanism  (NOT in repo)")
CACHE_DIR = MAR / "data/prostate158_cache"
TASK_A_OUT = Path("./experiments/prostate158_p2a")
OUT_DIR = Path(__file__).resolve().parent

# Add P2B to sys.path so we can import roi_utils
sys.path.insert(0, str(P2B))
from src.data.roi_utils import get_peri_lesional_ring, get_contralateral_mirror_roi  # type: ignore  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
DETECTION_THRESHOLD = 0.5
PERI_RING_MM = 5.0
SPACING_MM = (3.0, 0.5, 0.5)   # (D, H, W) physical spacing in mm
MIN_CC_VOXELS = 10              # minimum connected-component size to retain as FP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Low-level helpers ──────────────────────────────────────────────────────────

def compute_contrast_ratio(
    volume: np.ndarray,
    roi: np.ndarray,
    peri: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """(mean(volume[roi]) - mean(volume[peri])) / (|mean(volume[peri])| + eps).

    Returns float('nan') if either mask is empty.
    """
    if roi.sum() == 0 or peri.sum() == 0:
        return float("nan")
    m_roi = float(volume[roi.astype(bool)].mean())
    m_peri = float(volume[peri.astype(bool)].mean())
    return (m_roi - m_peri) / (abs(m_peri) + eps)


def _squeeze(x: object) -> np.ndarray:
    """Convert torch.Tensor or array-like to 3D numpy, dropping all leading size-1 dims."""
    arr = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def load_case(case_path: Path) -> dict:
    """Load a Prostate158 cache .pt file and return normalised dict.

    Returns
    -------
    dict with keys:
        t2w, hbv, adc  : float32 numpy, shape (32, 128, 128)
        label          : bool numpy, shape (32, 128, 128)  [GT lesion]
        gland_mask     : bool numpy, all-ones surrogate (no gland mask in cache)
        patient_id     : str
        is_positive    : bool
    """
    d = torch.load(case_path, weights_only=False)
    label = _squeeze(d["label"])
    # label is float32 in cache (0.0 / 1.0); binarise
    label_bool = label > 0.5
    out = {
        "t2w": _squeeze(d["t2w"]),
        "hbv": _squeeze(d["hbv"]),
        "adc": _squeeze(d["adc"]),
        "label": label_bool,
        "patient_id": d["patient_id"],
        "is_positive": bool(d.get("is_positive", label_bool.any())),
        # Full-volume gland surrogate — Prostate158 cache has no gland_mask key
        "gland_mask": np.ones(label_bool.shape, dtype=bool),
    }
    return out


def get_fp_mask(
    detection: np.ndarray,
    gt: np.ndarray,
    threshold: float = DETECTION_THRESHOLD,
    min_voxels: int = MIN_CC_VOXELS,
) -> np.ndarray:
    """FP = (detection > threshold) AND NOT GT, filtered to CCs >= min_voxels.

    Returns bool mask, same shape as detection.
    """
    fp_raw = (detection > threshold) & (~gt)
    if fp_raw.sum() == 0:
        return fp_raw
    labeled, n = ndimage.label(fp_raw)
    if n == 0:
        return fp_raw
    sizes = ndimage.sum(fp_raw, labeled, range(1, n + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    if not keep_labels:
        return np.zeros_like(fp_raw, dtype=bool)
    return np.isin(labeled, keep_labels)


# ── Per-case processing ────────────────────────────────────────────────────────

def process_case(case: dict, detection: np.ndarray) -> dict:
    """Compute contrast ratios for one case.

    Parameters
    ----------
    case : dict
        Output of load_case().
    detection : np.ndarray
        P1 detection probability map, shape (32, 128, 128), float32.

    Returns
    -------
    dict with keys: id, has_gt, fp_voxels, benign_voxels,
                    {t2w|hbv|adc}_{gt|fp|benign}
    """
    gt = case["label"]          # bool (32, 128, 128)
    gland = case["gland_mask"]  # bool (32, 128, 128), all-ones surrogate

    fp = get_fp_mask(detection, gt)
    # Contralateral benign = mirror of GT across W midline
    # Returns zeros if no GT or if bilateral overlap > 50%
    benign = get_contralateral_mirror_roi(
        gt.astype(np.uint8), gland.astype(np.uint8), spacing_mm=SPACING_MM
    ).astype(bool)

    out = {
        "id": case["patient_id"],
        "has_gt": bool(gt.any()),
        "fp_voxels": int(fp.sum()),
        "benign_voxels": int(benign.sum()),
    }

    # Build peri-lesional rings (empty mask → nan contrast)
    if gt.any():
        peri_gt = get_peri_lesional_ring(
            gt.astype(np.uint8), gland.astype(np.uint8),
            dilation_mm=PERI_RING_MM, spacing_mm=SPACING_MM,
        ).astype(bool)
    else:
        peri_gt = np.zeros_like(gt, dtype=bool)

    if fp.any():
        peri_fp = get_peri_lesional_ring(
            fp.astype(np.uint8), gland.astype(np.uint8),
            dilation_mm=PERI_RING_MM, spacing_mm=SPACING_MM,
        ).astype(bool)
    else:
        peri_fp = np.zeros_like(gt, dtype=bool)

    if benign.any():
        peri_benign = get_peri_lesional_ring(
            benign.astype(np.uint8), gland.astype(np.uint8),
            dilation_mm=PERI_RING_MM, spacing_mm=SPACING_MM,
        ).astype(bool)
    else:
        peri_benign = np.zeros_like(gt, dtype=bool)

    for mod_name, vol in (("t2w", case["t2w"]), ("hbv", case["hbv"]), ("adc", case["adc"])):
        # GT contrast: only computed for positive cases
        out[f"{mod_name}_gt"] = (
            compute_contrast_ratio(vol, gt, peri_gt) if gt.any() else float("nan")
        )
        # FP contrast: computed whenever FP exists (positive or negative cases)
        out[f"{mod_name}_fp"] = (
            compute_contrast_ratio(vol, fp, peri_fp) if fp.any() else float("nan")
        )
        # Benign contrast: only for positive cases with a valid mirror ROI
        out[f"{mod_name}_benign"] = (
            compute_contrast_ratio(vol, benign, peri_benign) if benign.any() else float("nan")
        )

    return out


# ── Aggregation ────────────────────────────────────────────────────────────────

def summarize(per_case: list) -> dict:
    """Aggregate ratio keys across all cases (NaN-safe).

    Returns dict: {key: {mean, sd, median, iqr, n}} for each *_{gt|fp|benign} field.

    Note on T2W contrast instability:
        The ratio formula (mean_roi - mean_peri) / (|mean_peri| + eps) can blow up
        when peri-ring mean is near zero (common in T2W for large FP blobs spanning
        the whole prostate). Use the median/IQR for T2W reporting; ADC is stable.
    """
    if not per_case:
        return {}
    suffix_keys = [k for k in per_case[0] if k.endswith(("_gt", "_fp", "_benign"))]
    out = {}
    for k in suffix_keys:
        vals = [
            c[k]
            for c in per_case
            if isinstance(c.get(k), float) and not np.isnan(c[k])
        ]
        if vals:
            arr = np.array(vals)
            out[k] = {
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                "n": len(arr),
            }
        else:
            out[k] = {"mean": None, "sd": None, "median": None, "iqr": None, "n": 0}
    return out


# ── Seed runner ────────────────────────────────────────────────────────────────

def run_seed(seed: int) -> dict:
    """Run the full pipeline for one seed.  Returns result dict."""
    det_path = TASK_A_OUT / f"detections_seed{seed}.npz"
    if not det_path.exists():
        raise FileNotFoundError(
            f"Task A detection sidecar not found: {det_path}\n"
            f"Run eval_prostate158_refinement.py for seed {seed} first."
        )
    logger.info("Loading detections: %s", det_path)
    detections = np.load(det_path)
    logger.info("  %d entries in detection sidecar", len(detections.files))

    per_case = []
    n_skipped = 0
    case_files = sorted(CACHE_DIR.glob("*.pt"))
    logger.info("Processing %d cache files ...", len(case_files))

    for cf in case_files:
        case = load_case(cf)
        pid = case["patient_id"]

        if pid not in detections.files:
            logger.warning("No detection for %s — skipping", pid)
            n_skipped += 1
            continue

        det = detections[pid]
        # Squeeze any singleton dims (det should already be (32,128,128))
        det = np.squeeze(det)
        if det.shape != case["label"].shape:
            logger.warning(
                "Detection shape %s != label shape %s for %s — skipping",
                det.shape, case["label"].shape, pid,
            )
            n_skipped += 1
            continue

        result = process_case(case, det)
        per_case.append(result)

    logger.info(
        "Processed %d cases (%d skipped)", len(per_case), n_skipped
    )
    summary = summarize(per_case)
    return {
        "seed": seed,
        "n_cases": len(per_case),
        "n_skipped": n_skipped,
        "per_case": per_case,
        "summary": summary,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute FP-mimic contrast ratios on Prostate158 (P2b analysis)."
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to process (default: 42 123 456 789 1024)",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-run even if output JSON already exists",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: dict = {}

    for seed in args.seeds:
        out_path = OUT_DIR / f"fp_mimic_seed{seed}.json"
        if out_path.exists() and not args.overwrite:
            logger.info("Skipping seed %d (%s exists). Use --overwrite to re-run.", seed, out_path)
            # Still load summary for aggregate output
            with out_path.open() as f:
                cached = json.load(f)
            all_summaries[f"seed{seed}"] = cached.get("summary", {})
            continue

        logger.info("=== Seed %d ===", seed)
        result = run_seed(seed)

        with out_path.open("w") as f:
            json.dump(result, f, indent=2, default=float)
        logger.info("Saved: %s", out_path)

        s = result["summary"]
        t2w_gt   = s.get("t2w_gt",     {}).get("mean")
        t2w_fp   = s.get("t2w_fp",     {}).get("mean")
        t2w_bn   = s.get("t2w_benign", {}).get("mean")
        adc_gt   = s.get("adc_gt",     {}).get("mean")
        adc_fp   = s.get("adc_fp",     {}).get("mean")
        adc_bn   = s.get("adc_benign", {}).get("mean")

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  nan"

        logger.info(
            "Seed %d summary: T2W gt=%s fp=%s benign=%s | ADC gt=%s fp=%s benign=%s",
            seed,
            _fmt(t2w_gt), _fmt(t2w_fp), _fmt(t2w_bn),
            _fmt(adc_gt), _fmt(adc_fp), _fmt(adc_bn),
        )
        all_summaries[f"seed{seed}"] = s

    # Save cross-seed aggregate
    agg_path = OUT_DIR / "fp_mimic_aggregate.json"
    with agg_path.open("w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Cross-seed aggregate saved: %s", agg_path)

    # Pretty-print summary table to stdout
    print("\n" + "=" * 110)
    print("FP-mimic contrast ratio summary — Prostate158  (mean [median] ± SD)")
    print(f"{'seed':<8} {'roi':<8} {'T2W mean[med]':>20} {'ADC mean[med]':>20} {'HBV mean[med]':>20}  n")
    print("-" * 110)
    for seed_key, s in all_summaries.items():
        for roi in ("gt", "fp", "benign"):
            t2 = s.get(f"t2w_{roi}", {})
            ad = s.get(f"adc_{roi}", {})
            hb = s.get(f"hbv_{roi}", {})

            def _cell(d):
                if d.get("mean") is None:
                    return "               nan"
                med = d.get("median")
                med_str = f"{med:+.4f}" if med is not None else "nan"
                return f"{d['mean']:+.4f}[{med_str}]±{d.get('sd', 0):.3f}"

            print(f"{seed_key:<8} {roi:<8} {_cell(t2):>26} {_cell(ad):>26} {_cell(hb):>26}  {t2.get('n', 0)}")
    print("=" * 110)
    print("NOTE: T2W mean is skewed by near-zero peri-ring cases; use median for T2W.")


if __name__ == "__main__":
    main()
