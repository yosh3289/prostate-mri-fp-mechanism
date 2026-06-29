#!/usr/bin/env python3
"""Compute 5-seed cross-seed aggregate for P2b fp_mimic analysis.
GT and benign are the same across seeds (same GT labels, same contralateral mirror).
FP differs by seed (different detection outputs).
Uses the 4-seed aggregate (123,456,789,1024) for benign/GT; seed42 FP is included separately.
"""
import json
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent

# Read aggregate (4 seeds: 123,456,789,1024)
with open(here / "fp_mimic_aggregate.json") as f:
    agg = json.load(f)

# Read seed42 per-case data for FP metrics
with open(here / "fp_mimic_seed42.json") as f:
    d42 = json.load(f)

cases42 = d42["per_case"]
adc_fp_42 = [c["adc_fp"] for c in cases42 if not np.isnan(c.get("adc_fp", float("nan")))]
t2w_fp_42 = [c["t2w_fp"] for c in cases42 if not np.isnan(c.get("t2w_fp", float("nan")))]

print("Seed42 FP stats:")
print("  adc_fp: n=%d mean=%.4f sd=%.4f median=%.4f" % (
    len(adc_fp_42), np.mean(adc_fp_42), np.std(adc_fp_42, ddof=1), np.median(adc_fp_42)))
print("  t2w_fp: n=%d median=%.4f mean=%.4f sd=%.4f" % (
    len(t2w_fp_42), np.median(t2w_fp_42), np.mean(t2w_fp_42), np.std(t2w_fp_42, ddof=1)))

print()
print("4-seed aggregate (GT/benign common across seeds):")
# GT and benign are the same (from aggregate which is per-case, same GT/labels)
# Just use seed123 as representative (all have same GT/benign)
s123 = agg["seed123"]
print("  t2w_gt median=%.4f mean=%.4f" % (s123["t2w_gt"]["median"], s123["t2w_gt"]["mean"]))
print("  t2w_benign median=%.4f mean=%.4f" % (s123["t2w_benign"]["median"], s123["t2w_benign"]["mean"]))
print("  adc_gt mean=%.4f sd=%.4f" % (s123["adc_gt"]["mean"], s123["adc_gt"]["sd"]))
print("  adc_benign mean=%.4f sd=%.4f" % (s123["adc_benign"]["mean"], s123["adc_benign"]["sd"]))

print()
print("FP values per seed:")
seeds_in_agg = ["seed123", "seed456", "seed789", "seed1024"]
for sk in seeds_in_agg:
    s = agg[sk]
    print("  %s: t2w_fp median=%.4f, adc_fp mean=%.4f sd=%.4f" % (
        sk, s["t2w_fp"]["median"], s["adc_fp"]["mean"], s["adc_fp"]["sd"]))
print("  seed42: t2w_fp median=%.4f, adc_fp mean=%.4f sd=%.4f" % (
    np.median(t2w_fp_42), np.mean(adc_fp_42), np.std(adc_fp_42, ddof=1)))

# 5-seed FP adc mean: average of per-seed means
adc_fp_means = [agg[sk]["adc_fp"]["mean"] for sk in seeds_in_agg] + [np.mean(adc_fp_42)]
adc_fp_sds = [agg[sk]["adc_fp"]["sd"] for sk in seeds_in_agg] + [np.std(adc_fp_42, ddof=1)]
t2w_fp_medians = [agg[sk]["t2w_fp"]["median"] for sk in seeds_in_agg] + [np.median(t2w_fp_42)]

print()
print("5-seed cross-seed aggregate:")
print("  adc_fp mean-of-means=%.4f sd-of-means=%.4f" % (np.mean(adc_fp_means), np.std(adc_fp_means, ddof=1)))
print("  t2w_fp median-of-medians=%.4f" % np.median(t2w_fp_medians))
print("  t2w_fp per-seed medians:", [round(m, 4) for m in t2w_fp_medians])
print("  adc_fp per-seed means:", [round(m, 4) for m in adc_fp_means])
print()
print("GT (all seeds identical):")
print("  t2w_gt median=%.4f" % s123["t2w_gt"]["median"])
print("  adc_gt mean=%.4f sd=%.4f" % (s123["adc_gt"]["mean"], s123["adc_gt"]["sd"]))
print()
print("Benign (all seeds identical):")
print("  t2w_benign median=%.4f" % s123["t2w_benign"]["median"])
print("  adc_benign mean=%.4f sd=%.4f" % (s123["adc_benign"]["mean"], s123["adc_benign"]["sd"]))
