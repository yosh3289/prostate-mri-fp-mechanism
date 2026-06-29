#!/usr/bin/env python3
"""Compute seed 42 summary stats for fp_mimic analysis."""
import json
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent

with open(here / "fp_mimic_seed42.json") as f:
    d42 = json.load(f)

cases = d42["per_case"]
gt_cases = [c for c in cases if c.get("has_gt", False)]
all_cases = cases

t2w_gt = [c["t2w_gt"] for c in gt_cases]
t2w_fp = [c["t2w_fp"] for c in all_cases]
t2w_benign = [c["t2w_benign"] for c in all_cases]
adc_gt = [c["adc_gt"] for c in gt_cases]
adc_fp = [c["adc_fp"] for c in all_cases]
adc_benign = [c["adc_benign"] for c in all_cases]

print("n_cases=%d n_skipped=%d n_gt_cases=%d" % (len(cases), d42["n_skipped"], len(gt_cases)))
print("t2w_gt  n=%d median=%.4f mean=%.4f sd=%.4f" % (len(t2w_gt), np.median(t2w_gt), np.mean(t2w_gt), np.std(t2w_gt, ddof=1)))
print("t2w_fp  n=%d median=%.4f mean=%.4f sd=%.4f" % (len(t2w_fp), np.median(t2w_fp), np.mean(t2w_fp), np.std(t2w_fp, ddof=1)))
print("t2w_benign n=%d median=%.4f mean=%.4f sd=%.4f" % (len(t2w_benign), np.median(t2w_benign), np.mean(t2w_benign), np.std(t2w_benign, ddof=1)))
print("adc_gt  n=%d mean=%.4f sd=%.4f" % (len(adc_gt), np.mean(adc_gt), np.std(adc_gt, ddof=1)))
print("adc_fp  n=%d mean=%.4f sd=%.4f" % (len(adc_fp), np.mean(adc_fp), np.std(adc_fp, ddof=1)))
print("adc_benign n=%d mean=%.4f sd=%.4f" % (len(adc_benign), np.mean(adc_benign), np.std(adc_benign, ddof=1)))

seed42_summary = {
    "t2w_gt": {
        "mean": float(np.mean(t2w_gt)),
        "sd": float(np.std(t2w_gt, ddof=1)),
        "median": float(np.median(t2w_gt)),
        "n": len(t2w_gt),
    },
    "t2w_fp": {
        "mean": float(np.mean(t2w_fp)),
        "sd": float(np.std(t2w_fp, ddof=1)),
        "median": float(np.median(t2w_fp)),
        "n": len(t2w_fp),
    },
    "t2w_benign": {
        "mean": float(np.mean(t2w_benign)),
        "sd": float(np.std(t2w_benign, ddof=1)),
        "median": float(np.median(t2w_benign)),
        "n": len(t2w_benign),
    },
    "adc_gt": {
        "mean": float(np.mean(adc_gt)),
        "sd": float(np.std(adc_gt, ddof=1)),
        "median": float(np.median(adc_gt)),
        "n": len(adc_gt),
    },
    "adc_fp": {
        "mean": float(np.mean(adc_fp)),
        "sd": float(np.std(adc_fp, ddof=1)),
        "median": float(np.median(adc_fp)),
        "n": len(adc_fp),
    },
    "adc_benign": {
        "mean": float(np.mean(adc_benign)),
        "sd": float(np.std(adc_benign, ddof=1)),
        "median": float(np.median(adc_benign)),
        "n": len(adc_benign),
    },
}
print("SEED42_SUMMARY:", json.dumps(seed42_summary))
