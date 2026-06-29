#!/usr/bin/env python3
"""External validation of bare A2 backbone (no refinement head) on Prostate158.

Loads the P1-MAR A2 adaptive_nnunet checkpoint for each seed, runs inference
over all 158 cached Prostate158 cases, and reports PI-CAI metrics
(ranking_score / AUROC / AP) plus traditional (Dice / CaseSens / CaseSpec).

Outputs per seed:
  experiments/prostate158_p2a_bare/results_seed{SEED}.json   -- metrics dict
  experiments/prostate158_p2a_bare/detections_seed{SEED}.npz -- per-case softmax maps

Usage:
  conda run -n mar python experiments/prostate158_p2a_bare/eval_prostate158_bare.py --seeds 42
  conda run -n mar python experiments/prostate158_p2a_bare/eval_prostate158_bare.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MAR = Path("ADJUST_PATH/P1-MAR  (NOT in repo: detection backbone repo)")
CACHE_DIR = MAR / "data/prostate158_cache"
OUT_DIR = Path(__file__).resolve().parent

P1_CKPT_TMPL = str(
    MAR / "outputs/checkpoints/ablation_nnunet/A2/seed{seed}/adaptive_nnunet/best_model.pth"
)

SEEDS = [42, 123, 456, 789, 1024]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import from P1-MAR
# ---------------------------------------------------------------------------
if str(MAR) not in sys.path:
    sys.path.insert(0, str(MAR))

from src.models.adaptive_native import build_adaptive_native  # type: ignore[import]
from src.utils.metrics import MetricTracker                    # type: ignore[import]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Prostate158CacheDataset(Dataset):
    """Reads pre-cached .pt files (identical field format to PI-CAI training)."""

    def __init__(self, cache_dir: str | Path):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {cache_dir}")
        logger.info("Prostate158CacheDataset: found %d cases in %s", len(self.files), cache_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], weights_only=False)
        return {
            "t2w": data["t2w"],
            "hbv": data["hbv"],
            "adc": data["adc"],
            "label": data["label"],
            "patient_id": data["patient_id"],
            "is_positive": data["is_positive"],
        }


# ---------------------------------------------------------------------------
# PI-CAI metric wrapper
# ---------------------------------------------------------------------------
def compute_picai_metrics(
    y_det: list[np.ndarray],
    y_true: list[np.ndarray],
    subject_list: list[str] | None = None,
) -> dict[str, float]:
    try:
        from picai_eval import evaluate
        from report_guided_annotation import extract_lesion_candidates

        detection_maps = [extract_lesion_candidates(s)[0] for s in y_det]
        metrics = evaluate(y_det=detection_maps, y_true=y_true, subject_list=subject_list)
        return {
            "auroc": float(metrics.auroc),
            "ap": float(metrics.AP),
            "ranking_score": float(metrics.score),
        }
    except Exception as exc:
        logger.warning("picai_eval failed: %s", exc)
        return {"auroc": float("nan"), "ap": float("nan"), "ranking_score": float("nan")}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_bare_model(seed: int, device: torch.device) -> torch.nn.Module:
    """Load P1-MAR A2 backbone (bare, no refinement head)."""
    ckpt_path = Path(P1_CKPT_TMPL.format(seed=seed))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"P1 backbone ckpt not found: {ckpt_path}")

    config = {
        "model": {
            "num_classes": 2,
            "num_modalities": 3,
            "nnunet_base_features": 32,
            "use_adaptive_gating": True,
        },
        "data": {"patch_crop_size": [64, 64, 16]},
    }
    model = build_adaptive_native("adaptive_nnunet", config)
    logger.info("Loading bare A2 ckpt: %s", ckpt_path)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    logger.info("Bare A2 model loaded (device=%s)", device)
    return model


# ---------------------------------------------------------------------------
# Inference loop for one seed
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_seed(
    seed: int,
    device: torch.device,
    cache_dir: Path,
    batch_size: int = 1,
) -> dict:
    model = load_bare_model(seed, device)
    ds = Prostate158CacheDataset(str(cache_dir))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    tracker = MetricTracker()
    all_y_det: list[np.ndarray] = []
    all_y_true: list[np.ndarray] = []
    all_ids: list[str] = []
    per_case_detections: dict[str, np.ndarray] = {}

    for batch_idx, batch in enumerate(loader):
        t2w   = batch["t2w"].to(device, non_blocking=True)
        hbv   = batch["hbv"].to(device, non_blocking=True)
        adc   = batch["adc"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with autocast():
            outputs = model([t2w, hbv, adc])
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if logits.shape[2:] != label.shape[2:]:
                logits = F.interpolate(
                    logits, size=label.shape[2:],
                    mode="trilinear", align_corners=False,
                )

        tracker.update(logits.float(), label)

        softmax_maps = torch.softmax(logits.float(), dim=1)[:, 1]  # [B, D, H, W]
        label_np = label[:, 0].cpu().numpy()

        for i in range(softmax_maps.shape[0]):
            det = softmax_maps[i].cpu().numpy().astype(np.float32)
            pid = batch["patient_id"][i]
            gt  = (label_np[i] >= 1).astype(np.int32)
            all_y_det.append(det)
            all_y_true.append(gt)
            all_ids.append(pid)
            per_case_detections[pid] = det

        if (batch_idx + 1) % 20 == 0:
            logger.info("  Processed %d/%d cases...", (batch_idx + 1) * batch_size, len(ds))

    logger.info("Inference complete. Computing metrics over %d cases.", len(all_ids))

    traditional = tracker.compute()
    picai = compute_picai_metrics(all_y_det, all_y_true, all_ids)

    # Save sidecar detections (.npz) -- needed by matched-sens analysis (Task C)
    det_path = OUT_DIR / f"detections_seed{seed}.npz"
    np.savez_compressed(str(det_path), **per_case_detections)
    logger.info(
        "Saved detection maps sidecar: %s  (%d cases)", det_path, len(per_case_detections)
    )

    result = {
        "seed": seed,
        "dataset": "prostate158",
        "n_cases": len(ds),
        **traditional,
        **picai,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eval bare P1-MAR A2 backbone on Prostate158 external set."
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to evaluate (default: all 5).",
    )
    ap.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (default 1).",
    )
    ap.add_argument(
        "--cache-dir", type=str, default=str(CACHE_DIR),
        help="Path to prostate158_cache/ directory.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run even if output already exists.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Evaluating seeds: %s", args.seeds)

    all_results: dict[str, dict] = {}

    for seed in args.seeds:
        out_path = OUT_DIR / f"results_seed{seed}.json"
        det_path = OUT_DIR / f"detections_seed{seed}.npz"

        if out_path.exists() and det_path.exists() and not args.force:
            logger.info(
                "Skipping seed %d -- results + detections already exist (use --force to redo)",
                seed,
            )
            with out_path.open() as f:
                all_results[f"seed{seed}"] = json.load(f)
            continue

        logger.info("=== Seed %d ===", seed)
        try:
            result = run_seed(seed, device, Path(args.cache_dir), batch_size=args.batch_size)
        except Exception:
            logger.exception("Seed %d failed", seed)
            continue

        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            "Seed %d done:  ranking_score=%.4f  AUROC=%.4f  AP=%.4f  "
            "Dice=%.4f  CaseSens=%.4f  CaseSpec=%.4f",
            seed,
            result.get("ranking_score", float("nan")),
            result.get("auroc", float("nan")),
            result.get("ap", float("nan")),
            result.get("dice", float("nan")),
            result.get("case_sensitivity", float("nan")),
            result.get("case_specificity", float("nan")),
        )
        all_results[f"seed{seed}"] = result

        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate across seeds (mean ± ddof=1 SD)
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        numeric_keys = [
            k for k, v in next(iter(all_results.values())).items()
            if isinstance(v, (int, float))
        ]
        agg: dict = {}
        for k in numeric_keys:
            vals = [
                r[k] for r in all_results.values()
                if isinstance(r.get(k), (int, float)) and not np.isnan(float(r[k]))
            ]
            if vals:
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "std_ddof1": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                }
        agg["n_seeds"] = len(all_results)
        agg_path = OUT_DIR / "aggregate.json"
        with agg_path.open("w") as f:
            json.dump(agg, f, indent=2)
        logger.info("Wrote aggregate.json (%d seeds)", len(all_results))

        logger.info("=== Aggregate (%d seeds) ===", len(all_results))
        for k in ["ranking_score", "auroc", "ap", "dice", "case_sensitivity", "case_specificity"]:
            if k in agg:
                logger.info("  %-22s: %.4f ± %.4f", k, agg[k]["mean"], agg[k]["std_ddof1"])
    elif len(all_results) == 1:
        result = next(iter(all_results.values()))
        logger.info(
            "Single-seed result:  ranking_score=%.4f  AUROC=%.4f  AP=%.4f  "
            "Dice=%.4f  CaseSens=%.4f  CaseSpec=%.4f",
            result.get("ranking_score", float("nan")),
            result.get("auroc", float("nan")),
            result.get("ap", float("nan")),
            result.get("dice", float("nan")),
            result.get("case_sensitivity", float("nan")),
            result.get("case_specificity", float("nan")),
        )


if __name__ == "__main__":
    main()
