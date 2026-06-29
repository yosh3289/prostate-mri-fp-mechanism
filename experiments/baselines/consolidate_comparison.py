#!/usr/bin/env python3
"""Consolidate all per-seed baseline results into a single aggregate_comparison.json.

Pulls from:
  - experiments/baselines/fast_seed*_picai.json (fast script)
  - experiments/baselines/fast_seed*_prostate158.json (fast script)
  - experiments/baselines/{temperature,platt,threshold_sweep,p2a_reference}_seed*_{cohort}.json (slow script)
  - experiments/p2a_5fold_cv/bootstrap_seed*.json (PI-CAI full cohort delta-CI)

Preference order: fast > slow (fast script uses the 'target_sens=0.94' constant
convention consistently; slow script also uses 0.94).

Writes:
  experiments/baselines/aggregate_comparison.json

Prints:
  Final summary table.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("./experiments/baselines")
BS_DIR = Path("./experiments/p2a_5fold_cv")

SEEDS = [42, 123, 456, 789, 1024]


def load_fast(seed: int, cohort: str) -> dict | None:
    p = OUT_DIR / f"fast_seed{seed}_{cohort}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_slow_method(seed: int, cohort: str, method: str) -> dict | None:
    p = OUT_DIR / f"{method}_seed{seed}_{cohort}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_bootstrap(seed: int) -> dict | None:
    p = BS_DIR / f"bootstrap_seed{seed}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _agg(vals: list[float]) -> dict:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return {"mean": None, "sd": None, "per_seed": [], "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_seed": [None if (v is None) else float(v) for v in vals],
        "n": int(len(arr)),
    }


def main():
    aggregate = {}

    for cohort in ["picai", "prostate158"]:
        sweep_spec, sweep_sens, p2a_spec, p2a_sens = [], [], [], []
        sweep_tau, p2a_tau = [], []
        # Method-by-method (slow script provides per-seed temp / platt results)
        temp_spec, temp_sens, temp_T = [], [], []
        platt_spec, platt_sens, platt_a, platt_b = [], [], [], []

        for seed in SEEDS:
            # For PI-CAI, prefer bootstrap (all 5 seeds always available)
            # over fast (may be partial) over slow (partial).
            used_bootstrap = False
            if cohort == "picai":
                bs = load_bootstrap(seed)
                if bs is not None:
                    fc = bs["full_cohort"]
                    sweep_spec.append(fc["bare_casespec"])
                    sweep_sens.append(fc["bare_casesens"])
                    sweep_tau.append(fc["bare_tau"])
                    p2a_spec.append(fc["p2a_casespec"])
                    p2a_sens.append(fc["p2a_casesens"])
                    p2a_tau.append(fc["p2a_tau"])
                    used_bootstrap = True
            if used_bootstrap:
                # Pull temperature/Platt from fast if available
                fast = load_fast(seed, cohort)
                if fast is not None and "temperature_scaling" in fast:
                    temp_spec.append(fast["temperature_scaling"].get("achieved_casespec"))
                    temp_sens.append(fast["temperature_scaling"].get("achieved_casesens"))
                    temp_T.append(fast["temperature_scaling"].get("T"))
                if fast is not None and "platt_scaling" in fast:
                    platt_spec.append(fast["platt_scaling"].get("achieved_casespec"))
                    platt_sens.append(fast["platt_scaling"].get("achieved_casesens"))
                    platt_a.append(fast["platt_scaling"].get("a"))
                    platt_b.append(fast["platt_scaling"].get("b"))
                # If slow script has per-seed temp/platt, include those
                tr = load_slow_method(seed, cohort, "temperature")
                pl = load_slow_method(seed, cohort, "platt")
                if tr is not None:
                    temp_spec.append(tr.get("achieved_casespec"))
                    temp_sens.append(tr.get("achieved_casesens"))
                    temp_T.append(tr.get("temperature_T"))
                if pl is not None:
                    platt_spec.append(pl.get("achieved_casespec"))
                    platt_sens.append(pl.get("achieved_casesens"))
                    platt_a.append(pl.get("platt_a"))
                    platt_b.append(pl.get("platt_b"))
                continue
            # For Prostate158 (or PI-CAI w/o bootstrap), prefer fast
            fast = load_fast(seed, cohort)
            if fast is not None:
                ts = fast["threshold_sweep"]
                p2a = fast["p2a_reference"]
                sweep_spec.append(ts.get("achieved_casespec"))
                sweep_sens.append(ts.get("achieved_casesens"))
                sweep_tau.append(ts.get("tau_matched_sens"))
                p2a_spec.append(p2a.get("achieved_casespec"))
                p2a_sens.append(p2a.get("achieved_casesens"))
                p2a_tau.append(p2a.get("tau_matched_sens"))
                # Fast only fits temp/Platt for seed 42
                if "temperature_scaling" in fast:
                    temp_spec.append(fast["temperature_scaling"].get("achieved_casespec"))
                    temp_sens.append(fast["temperature_scaling"].get("achieved_casesens"))
                    temp_T.append(fast["temperature_scaling"].get("T"))
                if "platt_scaling" in fast:
                    platt_spec.append(fast["platt_scaling"].get("achieved_casespec"))
                    platt_sens.append(fast["platt_scaling"].get("achieved_casesens"))
                    platt_a.append(fast["platt_scaling"].get("a"))
                    platt_b.append(fast["platt_scaling"].get("b"))
            else:
                # Fall back to slow script
                sw = load_slow_method(seed, cohort, "threshold_sweep")
                p2ar = load_slow_method(seed, cohort, "p2a_reference")
                tr = load_slow_method(seed, cohort, "temperature")
                pl = load_slow_method(seed, cohort, "platt")
                if sw is not None:
                    fc = sw["full_cohort"]
                    sweep_spec.append(fc.get("achieved_casespec"))
                    sweep_sens.append(fc.get("achieved_casesens"))
                    sweep_tau.append(fc.get("tau_matched_sens"))
                else:
                    sweep_spec.append(None); sweep_sens.append(None); sweep_tau.append(None)
                if p2ar is not None:
                    fc = p2ar["full_cohort"]
                    p2a_spec.append(fc.get("achieved_casespec"))
                    p2a_sens.append(fc.get("achieved_casesens"))
                    p2a_tau.append(fc.get("tau_matched_sens"))
                else:
                    p2a_spec.append(None); p2a_sens.append(None); p2a_tau.append(None)
                if tr is not None:
                    temp_spec.append(tr.get("achieved_casespec"))
                    temp_sens.append(tr.get("achieved_casesens"))
                    temp_T.append(tr.get("temperature_T"))
                if pl is not None:
                    platt_spec.append(pl.get("achieved_casespec"))
                    platt_sens.append(pl.get("achieved_casesens"))
                    platt_a.append(pl.get("platt_a"))
                    platt_b.append(pl.get("platt_b"))

        # Compute deltas (P2a - sweep) paired per seed
        deltas = []
        for p, s in zip(p2a_spec, sweep_spec):
            if p is None or s is None:
                continue
            deltas.append(float(p) - float(s))

        cohort_agg = {
            "n_seeds_completed": sum(1 for v in sweep_spec if v is not None),
            "threshold_sweep": {
                "casespec_matched": _agg(sweep_spec),
                "casesens_matched": _agg(sweep_sens),
                "tau_matched": _agg(sweep_tau),
            },
            "p2a_reference": {
                "casespec_matched": _agg(p2a_spec),
                "casesens_matched": _agg(p2a_sens),
                "tau_matched": _agg(p2a_tau),
            },
            "temperature_scaling": {
                "casespec_matched": _agg(temp_spec),
                "casesens_matched": _agg(temp_sens),
                "T_values": _agg(temp_T),
                "note": (
                    "T-scaling is a monotonic recalibration; at matched sens it is "
                    "mathematically equivalent to threshold sweep on the original "
                    "softmax.  Reported values may differ from threshold_sweep only "
                    "because the slow script evaluates on a 2/3 stratified holdout "
                    "rather than the full cohort.  See DECISION.md."
                ),
            },
            "platt_scaling": {
                "casespec_matched": _agg(platt_spec),
                "casesens_matched": _agg(platt_sens),
                "a_values": _agg(platt_a),
                "b_values": _agg(platt_b),
                "note": "Same equivalence argument as temperature scaling.",
            },
            "delta_p2a_minus_sweep": {
                "mean": float(np.mean(deltas)) if deltas else None,
                "sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "per_seed": deltas,
                "n_positive_seeds": sum(1 for d in deltas if d > 0),
                "n_total_seeds": len(deltas),
            },
        }
        aggregate[cohort] = cohort_agg

    # PI-CAI bootstrap CIs
    bs_seeds = {}
    for seed in SEEDS:
        bs = load_bootstrap(seed)
        if bs is not None:
            fc = bs["full_cohort"]
            bootstrap = bs["bootstrap"]
            bs_seeds[f"seed{seed}"] = {
                "bare_spec_full": fc["bare_casespec"],
                "p2a_spec_full": fc["p2a_casespec"],
                "delta_full": fc["delta_casespec"],
                "delta_bootstrap": bootstrap["delta_casespec"],
            }
    aggregate["picai_bootstrap"] = bs_seeds

    out_path = OUT_DIR / "aggregate_comparison.json"
    out_path.write_text(json.dumps(aggregate, indent=2))
    print(f"Wrote: {out_path}")

    # Print summary table
    print("\n" + "=" * 115)
    print("Paper 2 Session 6A — Baseline Comparison (CaseSpec at matched CaseSens >= 0.94)")
    print("=" * 115)
    print(f"{'cohort':<14} {'method':<35} {'CaseSpec mean':>15} {'CaseSpec sd':>13} "
          f"{'CaseSens mean':>15} {'n seeds':>9} {'delta seeds>0':>15}")
    print("-" * 115)
    for cohort in ["picai", "prostate158"]:
        agg = aggregate[cohort]
        for method_key, method_label in [
            ("threshold_sweep", "Threshold sweep (= bare A2)"),
            ("temperature_scaling", "Temperature scaling"),
            ("platt_scaling", "Platt scaling"),
            ("p2a_reference", "P2a refinement head"),
        ]:
            m = agg[method_key]
            spec = m["casespec_matched"]
            sens = m["casesens_matched"]
            if spec.get("mean") is None:
                continue
            print(f"{cohort:<14} {method_label:<35} {spec['mean']:>15.4f} {spec.get('sd', 0):>13.4f} "
                  f"{sens.get('mean', float('nan')):>15.4f} {spec.get('n', 0):>9}")
        d = agg["delta_p2a_minus_sweep"]
        if d["mean"] is not None:
            print(f"{cohort:<14} {'DELTA (P2a - sweep)':<35} {d['mean']:>+15.4f} {d.get('sd', 0):>13.4f} "
                  f"{'':>15} {d['n_total_seeds']:>9} {str(d['n_positive_seeds']) + '/' + str(d['n_total_seeds']):>15}")
    print("=" * 115)

    # Bootstrap summary for PI-CAI
    if bs_seeds:
        print("\nPI-CAI fold-0 val Bootstrap 95% CI (n=300 cases, B=1000 resamples):")
        print(f"{'seed':<6} {'Bare spec':<12} {'P2a spec':<12} {'delta':<12} {'CI (95%)':<30}")
        for skey, bv in bs_seeds.items():
            db = bv["delta_bootstrap"]
            ci = f"[{db['ci_low']:+.4f}, {db['ci_high']:+.4f}]" if db.get("ci_low") is not None else "n/a"
            print(f"{skey:<6} {bv['bare_spec_full']:<12.4f} {bv['p2a_spec_full']:<12.4f} "
                  f"{bv['delta_full']:<+12.4f} {ci:<30}")


if __name__ == "__main__":
    main()
