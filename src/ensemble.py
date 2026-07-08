"""
ensemble.py
===========
Ensemble the Dynamic V3 and Static V3 models by blending their logits,
then jointly optimise (blend_weight, threshold) on the validation set.

"This is the cheapest performance boost. You already have both trained.
 Ensembling typically gives +0.5–1.5% AUROC with zero additional training."
                                                   — literature survey doc

Method
------
  logit_final = w * logit_dynamic + (1 - w) * logit_static

We use Optuna to search over (w, threshold) jointly on the dev set — a 2-D
search that typically runs in under 2 minutes on CPU.

Usage
-----
    python ensemble.py

    # or programmatically:
    from ensemble import run_ensemble
    results = run_ensemble(cfg=CONFIG)
"""

import os

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from dataset  import HatefulMemesDatasetV2
from model    import AdaptiveFusionModelV3, StaticFusionModelV2

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "val_parquet":       "../Data/validation-00000-of-00001-1508d9e5032c2c1f.parquet",
    "clip_model":        "openai/clip-vit-large-patch14",
    "batch_size":        16,
    "dynamic_ckpt":      "../checkpoints/best_model_dynamic_v3.pt",
    "static_ckpt":       "../checkpoints/best_model_static_v3.pt",
    "optuna_trials":     200,       # number of (w, threshold) combos to try
    "output_dir":        "../checkpoints",
    "ensemble_cfg_name": "ensemble_config.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _get_logits(model, loader, device, is_dynamic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Run a full validation pass and return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["label"]

        if is_dynamic:
            logits, *_ = model(input_ids, attention_mask, pixel_values)
        else:
            logits, _  = model(input_ids, attention_mask, pixel_values)

        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_logits), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble(cfg: dict = None) -> dict:
    cfg = cfg or CONFIG

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device : {device}")

    # ── validation set ────────────────────────────────────────────────────────
    processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
    val_ds    = HatefulMemesDatasetV2(cfg["val_parquet"], processor, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=int(os.environ.get("NUM_WORKERS", 2)),
        pin_memory=(device.type == "cuda"),
    )

    # ── load dynamic model ────────────────────────────────────────────────────
    dyn_ckpt = torch.load(cfg["dynamic_ckpt"], map_location=device)
    dyn_cfg  = dyn_ckpt.get("config", {})
    dyn_model = AdaptiveFusionModelV3(
        clip_model_name=dyn_cfg.get("clip_model", cfg["clip_model"]),
        unfreeze_layers=dyn_cfg.get("unfreeze_layers", 4),
        attn_dropout=dyn_cfg.get("attn_dropout", 0.2),
    ).to(device)
    dyn_model.load_state_dict(dyn_ckpt["model_state"])
    print(f"Loaded dynamic model  — val AUROC: {dyn_ckpt.get('val_auroc', '?'):.4f}")

    # ── load static model ─────────────────────────────────────────────────────
    sta_ckpt = torch.load(cfg["static_ckpt"], map_location=device)
    sta_cfg  = sta_ckpt.get("config", {})
    sta_model = StaticFusionModelV2(
        clip_model_name=sta_cfg.get("clip_model", cfg["clip_model"]),
        unfreeze_layers=sta_cfg.get("unfreeze_layers", 4),
        static_alpha=sta_cfg.get("static_alpha", 0.5),
        attn_dropout=sta_cfg.get("attn_dropout", 0.2),
    ).to(device)
    sta_model.load_state_dict(sta_ckpt["model_state"])
    print(f"Loaded static model   — val AUROC: {sta_ckpt.get('val_auroc', '?'):.4f}")

    # ── collect logits ────────────────────────────────────────────────────────
    print("\nCollecting validation logits …")
    dyn_logits, labels = _get_logits(dyn_model, val_loader, device, is_dynamic=True)
    sta_logits, _      = _get_logits(sta_model, val_loader, device, is_dynamic=False)

    # ── individual metrics ────────────────────────────────────────────────────
    for name, logits in [("Dynamic V3", dyn_logits), ("Static V3",  sta_logits)]:
        probs  = 1 / (1 + np.exp(-logits))   # sigmoid
        auroc  = roc_auc_score(labels, probs)
        acc    = accuracy_score(labels, (probs >= 0.5).astype(int))
        print(f"  {name:12s} → AUROC: {auroc:.4f}  Acc@0.5: {acc:.4f}")

    # ── Optuna: jointly optimise (w, threshold) ───────────────────────────────
    print(f"\nOptuna search over (blend_weight, threshold) — {cfg['optuna_trials']} trials …")

    def objective(trial):
        w     = trial.suggest_float("w",         0.1, 0.9)
        thr   = trial.suggest_float("threshold", 0.1, 0.9)
        blend = w * dyn_logits + (1 - w) * sta_logits
        probs = 1 / (1 + np.exp(-blend))
        preds = (probs >= thr).astype(int)
        # Maximise macro-F1 (or swap for AUROC if preferred)
        return f1_score(labels, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["optuna_trials"], show_progress_bar=False)

    best_w   = study.best_params["w"]
    best_thr = study.best_params["threshold"]

    # ── final ensemble metrics ────────────────────────────────────────────────
    blend_logits = best_w * dyn_logits + (1 - best_w) * sta_logits
    blend_probs  = 1 / (1 + np.exp(-blend_logits))
    preds_opt    = (blend_probs >= best_thr).astype(int)
    preds_05     = (blend_probs >= 0.5).astype(int)

    results = {
        "blend_weight":    best_w,
        "threshold":       best_thr,
        "auroc":           roc_auc_score(labels, blend_probs),
        "accuracy_opt":    accuracy_score(labels, preds_opt),
        "accuracy_05":     accuracy_score(labels, preds_05),
        "f1_opt":          f1_score(labels, preds_opt, average="macro"),
        "f1_05":           f1_score(labels, preds_05,  average="macro"),
    }

    print(f"\n{'='*60}")
    print("Ensemble Results")
    print(f"{'='*60}")
    print(f"Blend weight  : {best_w:.3f} (dynamic) / {1-best_w:.3f} (static)")
    print(f"Threshold     : {best_thr:.3f}")
    print(f"AUROC         : {results['auroc']:.4f}")
    print(f"Acc@opt       : {results['accuracy_opt']:.4f}")
    print(f"Acc@0.5       : {results['accuracy_05']:.4f}")
    print(f"F1@opt        : {results['f1_opt']:.4f}")
    print(f"{'='*60}")

    # ── save config ───────────────────────────────────────────────────────────
    os.makedirs(cfg["output_dir"], exist_ok=True)
    save_path = os.path.join(cfg["output_dir"], cfg["ensemble_cfg_name"])
    torch.save(results, save_path)
    print(f"Ensemble config saved → {save_path}")

    return results


if __name__ == "__main__":
    run_ensemble()
