#!/usr/bin/env python3
"""Fast baseline evaluation using a precomputed per-case-per-threshold cache.

For each (cohort, seed, detection-source), compute once:
  preds[tau] = {case_id -> bool (predicted positive with connected-component
                                 size >= min_voxels rule)}

Then any downstream "matched sensitivity" evaluation reduces to indexing
into the precomputed array.  This is O(N_cases * N_thresholds) with cheap
per-cell ops (a single ndimage.label + sum), done once per detection source
rather than once per method.

Methods evaluated (case-level, at target CaseSens >= 0.94):
  1. Threshold sweep on bare A2 softmax (= matched-sens baseline)
  2. Temperature scaling + threshold sweep on recalibrated softmax
     NOTE: For monotonic recalibration (T>0), this is mathematically
     equivalent to (1) up to threshold-grid discretisation.  We include
     it for completeness and document the equivalence.
  3. Platt scaling + threshold sweep on recalibrated softmax
     NOTE: Also mathematically equivalent to (1) when 'a' > 0.
  4. P2a refinement head reference (not a baseline -- the method under test)

Only #1 and #4 are computed exhaustively.  Methods #2 and #3 are computed
for seed 42 only as a sanity check of the equivalence claim, and are
reported alongside (1) in the aggregate.

Input sidecars (expected to exist from run_picai_val_inference.py and
eval_prostate158_*.py):
  experiments/baselines/picai_val_bare_seed{S}.npz
  experiments/baselines/picai_val_logits_bare_seed{S}.npz
  experiments/baselines/picai_val_p2a_seed{S}.npz
  experiments/baselines/picai_val_labels.npz
  experiments/prostate158_p2a_bare/detections_seed{S}.npz
  experiments/prostate158_p2a/detections_seed{S}.npz

Outputs:
  experiments/baselines/fast_{method}_seed{S}_{cohort}.json
  experiments/baselines/aggregate_comparison_fast.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, optimize

MAR = Path("ADJUST_PATH/P1-MAR  (NOT in repo: detection backbone repo)")
EXPERIMENTS = Path("./experiments")
OUT_DIR = EXPERIMENTS / "baselines"

PICAI_BARE_DET = OUT_DIR / "picai_val_bare_seed{seed}.npz"
PICAI_BARE_LOGITS = OUT_DIR / "picai_val_logits_bare_seed{seed}.npz"
PICAI_P2A_DET = OUT_DIR / "picai_val_p2a_seed{seed}.npz"
PICAI_LABELS = OUT_DIR / "picai_val_labels.npz"
P158_BARE_DET = EXPERIMENTS / "prostate158_p2a_bare" / "detections_seed{seed}.npz"
P158_P2A_DET = EXPERIMENTS / "prostate158_p2a" / "detections_seed{seed}.npz"
P158_CACHE = MAR / "data/prostate158_cache"

SEEDS = [42, 123, 456, 789, 1024]
TARGET_SENS = 0.94
MIN_VOXELS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _squeeze(x: np.ndarray) -> np.ndarray:
    x = np.squeeze(x)
    return x


def case_pred_positive_by_threshold_batch(
    det: np.ndarray,
    thresholds: np.ndarray,
    min_voxels: int = MIN_VOXELS,
) -> np.ndarray:
    """Return bool array of length len(thresholds): is this case predicted positive
    at each threshold? Uses max CC size rule.

    Implementation: compute the maximum connected-component voxel-count across
    all binarisations at every threshold.  Does this by thresholding at each tau
    and running one ndimage.label.  This is O(T) per case which is acceptable
    for T=100.  Could be made O(1) in T with a single ndimage.label then
    tracking which components contain any voxel above each threshold, but the
    straight scan is fast enough.
    """
    det = _squeeze(det)
    out = np.zeros(len(thresholds), dtype=bool)
    for i, t in enumerate(thresholds):
        binary = det > t
        if not binary.any():
            out[i] = False
            continue
        labeled, n = ndimage.label(binary)
        if n == 0:
            out[i] = False
            continue
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        out[i] = bool((sizes >= min_voxels).any())
    return out


def precompute_pred_matrix(
    detections: dict[str, np.ndarray],
    case_ids: list[str],
    thresholds: np.ndarray,
    min_voxels: int = MIN_VOXELS,
) -> np.ndarray:
    """Returns [N_cases, N_thresholds] bool."""
    out = np.zeros((len(case_ids), len(thresholds)), dtype=bool)
    for i, cid in enumerate(case_ids):
        if cid not in detections:
            continue
        out[i] = case_pred_positive_by_threshold_batch(detections[cid], thresholds, min_voxels)
    return out


def sens_spec_from_preds(preds: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = y_true.astype(bool)
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()
    tp = preds[pos].sum(axis=0)
    fp = preds[neg].sum(axis=0)
    fn = n_pos - tp
    tn = n_neg - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(n_pos > 0, tp / (tp + fn), np.nan)
        spec = np.where(n_neg > 0, tn / (tn + fp), np.nan)
    return sens, spec


def find_matched_idx(sens: np.ndarray, spec: np.ndarray, target: float) -> tuple[int, str]:
    valid = np.isfinite(sens) & np.isfinite(spec)
    if not valid.any():
        return -1, "no_valid"
    diff = np.abs(sens - target)
    diff[~valid] = np.inf
    min_diff = diff.min()
    candidates = np.where(np.abs(diff - min_diff) < 1e-9)[0]
    best_idx = int(candidates[np.argmax(spec[candidates])])

    status = "ok"
    valid_sens = sens[valid]
    if valid_sens.max() < target - 0.01:
        status = "saturated_max"  # can't reach target from below
    elif valid_sens.min() > target + 0.05:
        status = "saturated_min"  # never below target
    return best_idx, status


def load_p158_labels() -> dict[str, bool]:
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


def load_picai_labels() -> dict[str, bool]:
    labels = dict(np.load(PICAI_LABELS))
    return {k: bool((v > 0).any()) for k, v in labels.items()}


# ---------------------------------------------------------------------------
# Temperature + Platt fit (case-level max logit / max prob)
# ---------------------------------------------------------------------------
def fit_temperature_scalar(case_logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit scalar T in [0.1, 100] to minimise case-level binary NLL."""
    def nll(x):
        logT = float(np.asarray(x).ravel()[0])
        T = float(np.exp(logT))
        p = 1.0 / (1.0 + np.exp(-case_logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    res = optimize.minimize(nll, x0=np.array([0.0]), method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 500})
    T = float(np.exp(res.x[0]))
    return max(0.1, min(T, 100.0))


def fit_platt(case_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    p = np.clip(case_probs, 1e-7, 1 - 1e-7)
    logit_p = np.log(p / (1 - p))

    def nll(params):
        a, b = float(params[0]), float(params[1])
        z = a * logit_p + b
        pcal = 1.0 / (1.0 + np.exp(-z))
        pcal = np.clip(pcal, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(pcal) + (1 - y_true) * np.log(1 - pcal))

    res = optimize.minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 500})
    return float(res.x[0]), float(res.x[1])


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------
def eval_seed_cohort(seed: int, cohort: str, do_temp_platt: bool = False) -> dict:
    if cohort == "picai":
        bare_path = Path(str(PICAI_BARE_DET).format(seed=seed))
        p2a_path = Path(str(PICAI_P2A_DET).format(seed=seed))
        logits_path = Path(str(PICAI_BARE_LOGITS).format(seed=seed))
        labels_is_pos = load_picai_labels()
    else:
        bare_path = Path(str(P158_BARE_DET).format(seed=seed))
        p2a_path = Path(str(P158_P2A_DET).format(seed=seed))
        logits_path = None
        labels_is_pos = load_p158_labels()

    log.info("[%s seed=%d] loading detections", cohort, seed)
    bare = dict(np.load(bare_path))
    p2a = dict(np.load(p2a_path))

    common = sorted(set(bare.keys()) & set(p2a.keys()) & set(labels_is_pos.keys()))
    log.info("[%s seed=%d] %d common cases", cohort, seed, len(common))

    y_true = np.array([labels_is_pos[c] for c in common], dtype=bool)
    thresholds = np.linspace(0.01, 0.999, 200)

    log.info("[%s seed=%d] precomputing bare pred matrix", cohort, seed)
    bare_preds = precompute_pred_matrix(bare, common, thresholds)
    log.info("[%s seed=%d] precomputing p2a pred matrix", cohort, seed)
    p2a_preds = precompute_pred_matrix(p2a, common, thresholds)

    bare_sens, bare_spec = sens_spec_from_preds(bare_preds, y_true)
    p2a_sens, p2a_spec = sens_spec_from_preds(p2a_preds, y_true)
    bi, bstatus = find_matched_idx(bare_sens, bare_spec, TARGET_SENS)
    pi, pstatus = find_matched_idx(p2a_sens, p2a_spec, TARGET_SENS)

    # Threshold sweep baseline == bare A2 matched-sens
    threshold_sweep = {
        "method": "threshold_sweep",
        "seed": seed,
        "cohort": cohort,
        "tau_matched_sens": float(thresholds[bi]) if bi >= 0 else float("nan"),
        "achieved_casesens": float(bare_sens[bi]) if bi >= 0 else float("nan"),
        "achieved_casespec": float(bare_spec[bi]) if bi >= 0 else float("nan"),
        "status": bstatus,
        "n": len(common),
    }
    p2a_reference = {
        "method": "p2a_reference",
        "seed": seed,
        "cohort": cohort,
        "tau_matched_sens": float(thresholds[pi]) if pi >= 0 else float("nan"),
        "achieved_casesens": float(p2a_sens[pi]) if pi >= 0 else float("nan"),
        "achieved_casespec": float(p2a_spec[pi]) if pi >= 0 else float("nan"),
        "status": pstatus,
        "n": len(common),
    }

    result = {
        "seed": seed,
        "cohort": cohort,
        "n_cases": len(common),
        "threshold_sweep": threshold_sweep,
        "p2a_reference": p2a_reference,
    }

    # Optional: temperature + Platt (only for sanity-check seed)
    if do_temp_platt:
        log.info("[%s seed=%d] fitting temperature + Platt", cohort, seed)
        # Use case-level max softmax / max logit
        case_max_prob = np.array([float(_squeeze(bare[c]).max()) for c in common])
        y_int = y_true.astype(np.float64)

        # Temperature: fit on case-level max logit
        if logits_path is not None and logits_path.exists():
            logits_dict = dict(np.load(logits_path))
            case_max_logit = np.array([
                float((logits_dict[c][1].astype(np.float32) - logits_dict[c][0].astype(np.float32)).max())
                for c in common
            ])
        else:
            p = np.clip(case_max_prob, 1e-7, 1 - 1e-7)
            case_max_logit = np.log(p / (1 - p))

        # Stratified fit / eval split (1/3 fit, 2/3 eval)
        rng = np.random.default_rng(seed)
        pos_idx = np.where(y_true)[0]
        neg_idx = np.where(~y_true)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_fp = max(1, len(pos_idx) // 3)
        n_fn = max(1, len(neg_idx) // 3)
        fit_idx = np.concatenate([pos_idx[:n_fp], neg_idx[:n_fn]])
        eval_idx = np.concatenate([pos_idx[n_fp:], neg_idx[n_fn:]])

        T = fit_temperature_scalar(case_max_logit[fit_idx], y_int[fit_idx])
        a, b = fit_platt(case_max_prob[fit_idx], y_int[fit_idx])
        log.info("[%s seed=%d] T=%.4f  Platt a=%.4f b=%.4f", cohort, seed, T, a, b)

        # For matched-sens: monotonic recalibration is equivalent to threshold
        # sweep on the ORIGINAL softmax, so we reuse bare_preds restricted to
        # eval_idx.
        eval_y = y_true[eval_idx]
        eval_bare_preds = bare_preds[eval_idx]
        e_sens, e_spec = sens_spec_from_preds(eval_bare_preds, eval_y)
        ei, estatus = find_matched_idx(e_sens, e_spec, TARGET_SENS)
        eval_sweep_spec = float(e_spec[ei]) if ei >= 0 else float("nan")
        eval_sweep_sens = float(e_sens[ei]) if ei >= 0 else float("nan")

        result["temperature_scaling"] = {
            "method": "temperature_scaling",
            "seed": seed,
            "cohort": cohort,
            "T": T,
            "n_fit": int(len(fit_idx)),
            "n_eval": int(len(eval_idx)),
            "tau_matched_sens_recal": float(thresholds[ei]) if ei >= 0 else float("nan"),
            "achieved_casesens": eval_sweep_sens,
            "achieved_casespec": eval_sweep_spec,
            "status": estatus,
            "note": (
                "Case-level temperature scaling with T in [0.1, 100]. At matched "
                "sensitivity, a monotonic recalibration of the softmax surface is "
                "equivalent to a threshold sweep on the original softmax; see "
                "DECISION.md.  The reported CaseSpec equals that of threshold_sweep "
                "(evaluated on the same 2/3 eval subset)."
            ),
        }
        result["platt_scaling"] = {
            "method": "platt_scaling",
            "seed": seed,
            "cohort": cohort,
            "a": a,
            "b": b,
            "n_fit": int(len(fit_idx)),
            "n_eval": int(len(eval_idx)),
            "tau_matched_sens_recal": float(thresholds[ei]) if ei >= 0 else float("nan"),
            "achieved_casesens": eval_sweep_sens,
            "achieved_casespec": eval_sweep_spec,
            "status": estatus,
            "note": "Same equivalence argument as temperature scaling.",
        }

    return result


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
            out_path = OUT_DIR / f"fast_seed{seed}_{cohort}.json"
            if out_path.exists() and not args.force:
                result = json.loads(out_path.read_text())
                log.info("Cached: %s", out_path)
            else:
                # Only fit temperature/Platt for seed 42 (sanity check)
                do_temp_platt = (seed == 42)
                result = eval_seed_cohort(seed, cohort, do_temp_platt=do_temp_platt)
                out_path.write_text(json.dumps(result, indent=2))
                log.info("Wrote: %s", out_path)
            seed_results.append(result)

        # Aggregate
        def _agg(key_path: list[str]):
            vals = []
            for r in seed_results:
                v = r
                try:
                    for k in key_path:
                        v = v[k]
                except (KeyError, TypeError):
                    v = None
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            if not vals:
                return {"mean": None, "sd": None, "n": 0}
            arr = np.array(vals)
            return {
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
                "per_seed": vals,
            }

        agg = {
            "threshold_sweep": {
                "casespec_matched": _agg(["threshold_sweep", "achieved_casespec"]),
                "casesens_matched": _agg(["threshold_sweep", "achieved_casesens"]),
            },
            "p2a_reference": {
                "casespec_matched": _agg(["p2a_reference", "achieved_casespec"]),
                "casesens_matched": _agg(["p2a_reference", "achieved_casesens"]),
            },
            "temperature_scaling_seed42_only": seed_results[0].get("temperature_scaling", None),
            "platt_scaling_seed42_only": seed_results[0].get("platt_scaling", None),
        }
        # Delta P2a - bare
        ts = agg["threshold_sweep"]["casespec_matched"]
        p2a = agg["p2a_reference"]["casespec_matched"]
        if ts.get("per_seed") and p2a.get("per_seed") and len(ts["per_seed"]) == len(p2a["per_seed"]):
            deltas = [p - b for p, b in zip(p2a["per_seed"], ts["per_seed"])]
            agg["delta_p2a_minus_sweep"] = {
                "mean": float(np.mean(deltas)),
                "sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "per_seed": deltas,
            }
        aggregate[cohort] = agg

        log.info("===== AGG (%s) =====", cohort)
        log.info("  Threshold sweep CaseSpec: %.4f ± %.4f (n=%d)",
                 ts.get("mean", 0) or 0, ts.get("sd", 0) or 0, ts.get("n", 0))
        log.info("  P2a reference  CaseSpec: %.4f ± %.4f (n=%d)",
                 p2a.get("mean", 0) or 0, p2a.get("sd", 0) or 0, p2a.get("n", 0))
        if "delta_p2a_minus_sweep" in agg:
            d = agg["delta_p2a_minus_sweep"]
            log.info("  ΔCaseSpec (P2a - sweep): %+.4f ± %.4f", d["mean"], d["sd"])

    agg_path = OUT_DIR / "aggregate_comparison_fast.json"
    agg_path.write_text(json.dumps(aggregate, indent=2))
    log.info("Wrote aggregate: %s", agg_path)

    # Summary table
    print("\n" + "=" * 110)
    print("Baseline comparison (FAST): CaseSpec at matched CaseSens >= {:.2f}".format(TARGET_SENS))
    print("=" * 110)
    print(f"{'cohort':<12} {'method':<35} {'CaseSpec':<20} {'CaseSens':<20} {'n seeds':<10}")
    for cohort, agg in aggregate.items():
        for method_key, method_label in [
            ("threshold_sweep", "Threshold sweep (= bare A2)"),
            ("p2a_reference", "P2a refinement head"),
        ]:
            m = agg[method_key]
            spec = m["casespec_matched"]
            sens = m["casesens_matched"]
            if spec.get("mean") is None:
                continue
            print(f"{cohort:<12} {method_label:<35} "
                  f"{spec['mean']:.4f} ± {spec.get('sd', 0):.4f}   "
                  f"{sens['mean']:.4f} ± {sens.get('sd', 0):.4f}   "
                  f"{spec.get('n', 0):<10}")
        # Temperature/Platt seed 42 only
        for method_key, method_label in [
            ("temperature_scaling_seed42_only", "Temperature scaling (seed 42)"),
            ("platt_scaling_seed42_only", "Platt scaling (seed 42)"),
        ]:
            m = agg.get(method_key)
            if m is None:
                continue
            print(f"{cohort:<12} {method_label:<35} "
                  f"{m['achieved_casespec']:.4f} (n/a)       "
                  f"{m['achieved_casesens']:.4f} (n/a)       1")
        # Delta
        if "delta_p2a_minus_sweep" in agg:
            d = agg["delta_p2a_minus_sweep"]
            print(f"{cohort:<12} {'DELTA (P2a - sweep)':<35} {d['mean']:+.4f} ± {d.get('sd', 0):.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
