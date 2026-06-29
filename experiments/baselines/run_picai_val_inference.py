#!/usr/bin/env python3
"""Generate per-case detection sidecars for PI-CAI fold-0 val (n=300).

For each seed, loads both:
  - Bare A2 backbone (P1-MAR ablation_nnunet/A2/seed{S}/adaptive_nnunet/best_model.pth)
  - P2a multi-scale v2 refinement head (fp_suppression_ms_v2_seed{S}/best_model.pth)

Runs inference on the 300 PI-CAI fold-0 val cases and saves two sidecars:
  - experiments/baselines/picai_val_bare_seed{S}.npz      -- softmax class-1 maps
  - experiments/baselines/picai_val_p2a_seed{S}.npz       -- P2a softmax class-1 maps
  - experiments/baselines/picai_val_logits_bare_seed{S}.npz -- raw class-0/class-1 logits
                                                            (needed for temperature/Platt scaling)
  - experiments/baselines/picai_val_labels.npz           -- GT label volumes (once, seed-independent)

Usage:
  conda run -n mar python experiments/baselines/run_picai_val_inference.py --seeds 42
  conda run -n mar python experiments/baselines/run_picai_val_inference.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import types
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
P2A = Path("ADJUST_PATH/P2-specificity-refinement  (NOT in repo)")
PICAI_CACHE = MAR / "data/cache"
FOLD0_SPLIT = MAR / "configs/fold0_split.json"
OUT_DIR = Path(__file__).resolve().parent

P1_CKPT_TMPL = str(MAR / "outputs/checkpoints/ablation_nnunet/A2/seed{seed}/adaptive_nnunet/best_model.pth")
P2A_CKPT_TMPL = str(P2A / "outputs/checkpoints/fp_suppression_ms_v2_seed{seed}/best_model.pth")

SEEDS = [42, 123, 456, 789, 1024]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import FPSuppressedMIGFNet with Windows-path patch (same approach as
# eval_prostate158_refinement.py)
# ---------------------------------------------------------------------------
def _load_fp_suppressed_migfnet_cls():
    src_path = P2A / "src/models/fp_suppressed_migfnet.py"
    src = src_path.read_text(encoding="utf-8")
    correct_native = str(MAR / "src/models/adaptive_native.py").replace("\\", "/")
    # The released backbone source hardcodes an absolute import path for its
    # adaptive_native module; set BACKBONE_NATIVE_IMPORT to match your copy so
    # it can be rewritten to the local backbone location. (Backbone not shipped.)
    BACKBONE_NATIVE_IMPORT = "ADJUST_PATH/adaptive_native.py"
    src = src.replace(BACKBONE_NATIVE_IMPORT, correct_native)
    mod = types.ModuleType("fp_suppressed_migfnet")
    mod.__file__ = str(src_path)
    exec(compile(src, str(src_path), "exec"), mod.__dict__)
    return mod.FPSuppressedMIGFNet


# Add MAR to sys.path for adaptive_native
if str(MAR) not in sys.path:
    sys.path.insert(0, str(MAR))

from src.models.adaptive_native import build_adaptive_native  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset: PI-CAI fold-0 val from cache
# ---------------------------------------------------------------------------
class PICAIFold0ValDataset(Dataset):
    def __init__(self, cache_dir: Path, split_json: Path):
        with split_json.open() as f:
            split = json.load(f)
        self.val_ids = split["val"]
        self.cache_dir = Path(cache_dir)
        # Verify files exist
        self.files = []
        for sid in self.val_ids:
            p = self.cache_dir / f"{sid}.pt"
            if not p.exists():
                log.warning("Missing cache file: %s", p)
                continue
            self.files.append(p)
        log.info("PICAI fold-0 val: %d cases (requested %d)", len(self.files), len(self.val_ids))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        # PI-CAI cache has shape [D,H,W]; add channel dim -> [1,D,H,W] to match
        # Prostate158 convention and model expected input.
        def _ensure_ch(x):
            if x.ndim == 3:
                return x.unsqueeze(0)
            return x
        return {
            "t2w": _ensure_ch(d["t2w"]),
            "hbv": _ensure_ch(d["hbv"]),
            "adc": _ensure_ch(d["adc"]),
            "label": _ensure_ch(d["label"]),
            "study_id": d["study_id"],
            "patient_id": d.get("patient_id", d["study_id"]),
        }


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_bare_a2(seed: int, device) -> torch.nn.Module:
    ckpt_path = Path(P1_CKPT_TMPL.format(seed=seed))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"P1 A2 ckpt missing: {ckpt_path}")
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
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_p2a(seed: int, device) -> torch.nn.Module:
    FPSuppressedMIGFNet = _load_fp_suppressed_migfnet_cls()
    p1_ckpt = Path(P1_CKPT_TMPL.format(seed=seed))
    p2a_ckpt_path = Path(P2A_CKPT_TMPL.format(seed=seed))
    if not p1_ckpt.exists():
        raise FileNotFoundError(f"P1 backbone ckpt missing: {p1_ckpt}")
    if not p2a_ckpt_path.exists():
        raise FileNotFoundError(f"P2a ckpt missing: {p2a_ckpt_path}")
    p2a_ckpt = torch.load(p2a_ckpt_path, map_location=device, weights_only=False)

    stored_ms = p2a_ckpt.get("multiscale")
    if stored_ms is None:
        rs = p2a_ckpt.get("refinement_state", {})
        use_ms = any("proj_dec" in k for k in rs.keys())
    else:
        use_ms = bool(stored_ms)

    model = FPSuppressedMIGFNet(multiscale=use_ms).to(device)
    model.load_p1_weights(str(p1_ckpt), device=str(device))

    if "refinement_state" in p2a_ckpt:
        model.load_state_dict(p2a_ckpt["refinement_state"], strict=False)
    elif "suppression_head_state" in p2a_ckpt:
        model.suppression_head.load_state_dict(p2a_ckpt["suppression_head_state"])
    else:
        raise KeyError(f"No refinement weights in ckpt: {list(p2a_ckpt.keys())}")

    return model.eval()


# ---------------------------------------------------------------------------
# Run inference for one seed
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device,
    save_logits: bool = False,
):
    """Returns (softmax_dict, logits_dict_or_None, labels_dict)."""
    softmax_out: dict[str, np.ndarray] = {}
    logits_out: dict[str, np.ndarray] | None = {} if save_logits else None
    labels_out: dict[str, np.ndarray] = {}

    for batch_idx, batch in enumerate(loader):
        t2w = batch["t2w"].to(device, non_blocking=True)
        hbv = batch["hbv"].to(device, non_blocking=True)
        adc = batch["adc"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with autocast():
            outputs = model([t2w, hbv, adc])
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if logits.shape[2:] != label.shape[2:]:
                logits = F.interpolate(
                    logits, size=label.shape[2:],
                    mode="trilinear", align_corners=False,
                )

        logits_f = logits.float()
        softmax_maps = torch.softmax(logits_f, dim=1)[:, 1]
        label_np = label[:, 0].cpu().numpy()

        for i in range(softmax_maps.shape[0]):
            sid = batch["study_id"][i]
            softmax_out[sid] = softmax_maps[i].cpu().numpy().astype(np.float32)
            if save_logits:
                # Save both class logits: [2, D, H, W] -> we only need class-1 for
                # scalar recalibration; for temperature scaling we actually need
                # both to recompute softmax.  Store as float16 to save space.
                logits_out[sid] = logits_f[i].cpu().numpy().astype(np.float16)
            labels_out[sid] = (label_np[i] >= 1).astype(np.uint8)

        if (batch_idx + 1) % 50 == 0:
            log.info("  %d / %d cases", batch_idx + 1, len(loader))

    return softmax_out, logits_out, labels_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--only-bare", action="store_true", help="Only run bare A2 inference")
    ap.add_argument("--only-p2a", action="store_true", help="Only run P2a inference")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    ds = PICAIFold0ValDataset(PICAI_CACHE, FOLD0_SPLIT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    labels_path = OUT_DIR / "picai_val_labels.npz"
    labels_saved = labels_path.exists()

    for seed in args.seeds:
        bare_soft = OUT_DIR / f"picai_val_bare_seed{seed}.npz"
        bare_logits = OUT_DIR / f"picai_val_logits_bare_seed{seed}.npz"
        p2a_soft = OUT_DIR / f"picai_val_p2a_seed{seed}.npz"

        # ---- Bare A2 ----
        if not args.only_p2a and (args.force or not (bare_soft.exists() and bare_logits.exists())):
            log.info("=== Seed %d: bare A2 ===", seed)
            model = load_bare_a2(seed, device)
            soft, logits, labels = run_inference(model, loader, device, save_logits=True)
            np.savez_compressed(str(bare_soft), **soft)
            np.savez_compressed(str(bare_logits), **logits)
            if not labels_saved:
                np.savez_compressed(str(labels_path), **labels)
                labels_saved = True
                log.info("Saved labels to %s", labels_path)
            del model, soft, logits, labels
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Saved: %s, %s", bare_soft, bare_logits)
        elif not args.only_p2a:
            log.info("Skipping bare A2 seed %d (exists)", seed)

        # ---- P2a ----
        if not args.only_bare and (args.force or not p2a_soft.exists()):
            log.info("=== Seed %d: P2a ===", seed)
            model = load_p2a(seed, device)
            soft, _, labels = run_inference(model, loader, device, save_logits=False)
            np.savez_compressed(str(p2a_soft), **soft)
            if not labels_saved:
                np.savez_compressed(str(labels_path), **labels)
                labels_saved = True
            del model, soft, labels
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Saved: %s", p2a_soft)
        elif not args.only_bare:
            log.info("Skipping P2a seed %d (exists)", seed)


if __name__ == "__main__":
    main()
