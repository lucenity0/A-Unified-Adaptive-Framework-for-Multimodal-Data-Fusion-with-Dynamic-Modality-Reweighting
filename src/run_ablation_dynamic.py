"""
run_ablation_dynamic.py
=======================
Trains ONLY the Dynamic Reweighting model (5 epochs) and prints
a full comparison table using saved Phase 1 ablation results.

No retraining of models 1-4 -- reads from ablation_results.txt.

Run:
    python3 run_ablation_dynamic.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from dataset import get_dataloaders
from model import AdaptiveFusionModel


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "train_parquet"  : "../Data/train-00000-of-00001-6587b3a58d350036.parquet",
    "val_parquet"    : "../Data/validation-00000-of-00001-1508d9e5032c2c1f.parquet",
    "batch_size"     : 16,
    "num_epochs"     : 5,
    "freeze_clip"    : True,
    "clip_lr"        : 1e-5,
    "fusion_lr"      : 3e-4,
    "weight_decay"   : 0.05,
    "pos_weight"     : 1.5,
    "entropy_weight" : 0.01,
    "results_dir"    : "../results",
    "phase1_results" : "../results/ablation_results.txt",
}

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────────────────────────
# READ PHASE 1 RESULTS FROM FILE
# ─────────────────────────────────────────────────────────────────
def load_phase1_results(path):
    """
    Parses ablation_results.txt and returns list of result dicts.
    Falls back to hardcoded values if file not found or unparseable.
    """
    fallback = [
        {"model": "1. Text Only",                                    "auroc": 0.6225, "acc": 0.5660, "f1": 0.5361},
        {"model": "2. Image Only",                                   "auroc": 0.6283, "acc": 0.5740, "f1": 0.5680},
        {"model": "3. Concat Fusion (no attention, no reweighting)", "auroc": 0.6707, "acc": 0.6020, "f1": 0.5907},
        {"model": "4. Cross-Attn (no reweighting)",                  "auroc": 0.6975, "acc": 0.6400, "f1": 0.6351},
        {"model": "5. Cross-Attn + Static Reweighting (Phase 1)",    "auroc": 0.7079, "acc": 0.6440, "f1": 0.6360},
    ]

    if not os.path.exists(path):
        print(f"Phase 1 results file not found at {path} -- using hardcoded values.")
        return fallback

    results = []
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-") or \
           line.startswith("Model") or line.startswith("ABLATION") or \
           line.startswith("Gain") or line.startswith("Key"):
            continue
        parts = line.rsplit(None, 4)
        if len(parts) == 5:
            try:
                results.append({
                    "model": parts[0],
                    "auroc": float(parts[1]),
                    "acc"  : float(parts[2]),
                    "f1"   : float(parts[3]),
                })
            except ValueError:
                continue

    if len(results) < 5:
        print(f"Only parsed {len(results)} rows -- using hardcoded fallback.")
        return fallback

    print(f"Loaded {len(results)} Phase 1 results from file.")
    return results


# ─────────────────────────────────────────────────────────────────
# TRAIN + EVAL DYNAMIC MODEL ONLY
# ─────────────────────────────────────────────────────────────────
def train_and_evaluate_dynamic(train_loader, val_loader):
    print(f"\n{'='*55}")
    print(f"  Training: 6. Cross-Attn + Dynamic Reweighting (Phase 2)")
    print(f"{'='*55}")

    model = AdaptiveFusionModel(freeze_clip=CONFIG["freeze_clip"]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(DEVICE)
    )

    optimizer = torch.optim.AdamW([
        {'params': model.clip.parameters(),       'lr': CONFIG["clip_lr"]},
        {'params': model.cross_attn.parameters(), 'lr': CONFIG["fusion_lr"]},
        {'params': model.gating.parameters(),     'lr': CONFIG["fusion_lr"]},
        {'params': model.classifier.parameters(), 'lr': CONFIG["fusion_lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    total_steps  = CONFIG["num_epochs"] * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_auroc = 0.0
    best_acc   = 0.0
    best_f1    = 0.0

    for epoch in range(CONFIG["num_epochs"]):

        # ── Train ─────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values   = batch['pixel_values'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits, alpha = model(input_ids, attention_mask, pixel_values)

            loss = criterion(logits, labels)

            # Entropy regularization -- forces alpha to be diverse
            eps = 1e-8
            alpha_entropy = -(alpha * torch.log(alpha + eps) +
                              (1 - alpha) * torch.log(1 - alpha + eps))
            entropy_loss  = -alpha_entropy.mean()
            loss = loss + CONFIG["entropy_weight"] * entropy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ── Evaluate ──────────────────────────────────────────
        model.eval()
        all_preds, all_labels, all_alphas = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values   = batch['pixel_values'].to(DEVICE)
                labels         = batch['label']

                logits, alpha = model(input_ids, attention_mask, pixel_values)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.numpy())
                all_alphas.extend(alpha.mean(dim=-1).cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_alphas = np.array(all_alphas)
        bin_preds  = (all_preds >= 0.5).astype(int)

        auroc = roc_auc_score(all_labels, all_preds)
        acc   = accuracy_score(all_labels, bin_preds)
        f1    = f1_score(all_labels, bin_preds, average='macro')

        print(f"  Epoch {epoch+1}/{CONFIG['num_epochs']} --> "
              f"AUROC: {auroc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Alpha Std: {all_alphas.std():.3f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_acc   = acc
            best_f1    = f1

    return {
        "model": "6. Cross-Attn + Dynamic Reweighting (Phase 2)",
        "auroc": best_auroc,
        "acc"  : best_acc,
        "f1"   : best_f1,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    print(f"Using device: {DEVICE}")

    # ── Load Phase 1 results ──────────────────────────────────────
    print(f"\nLoading Phase 1 results from {CONFIG['phase1_results']}...")
    phase1_results = load_phase1_results(CONFIG["phase1_results"])

    # ── Load data ─────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        CONFIG["train_parquet"],
        CONFIG["val_parquet"],
        batch_size=CONFIG["batch_size"]
    )

    # ── Train dynamic model only ──────────────────────────────────
    dynamic_result = train_and_evaluate_dynamic(train_loader, val_loader)

    # ── Combine all results ───────────────────────────────────────
    all_results    = phase1_results + [dynamic_result]
    baseline_auroc = all_results[0]["auroc"]

    # ── Print full table ──────────────────────────────────────────
    print("\n")
    print("=" * 74)
    print(f"{'FULL ABLATION STUDY -- Static + Dynamic Reweighting':^74}")
    print("=" * 74)
    print(f"{'Model':<51} {'AUROC':>7} {'Acc':>7} {'F1':>7} {'ΔAUROC':>8}")
    print("-" * 74)

    for r in all_results:
        delta     = r["auroc"] - baseline_auroc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        marker    = " *" if "Phase 2" in r["model"] else "  "
        print(f"{r['model']:<51}{marker} {r['auroc']:>7.4f} {r['acc']:>7.4f} "
              f"{r['f1']:>7.4f} {delta_str:>8}")

    print("=" * 74)
    print("  * Phase 2 -- Dynamic Reweighting (newly trained)")

    # ── Key finding ───────────────────────────────────────────────
    static_auroc  = phase1_results[4]["auroc"]
    dynamic_auroc = dynamic_result["auroc"]
    gain          = dynamic_auroc - static_auroc
    gain_str      = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"

    print(f"\nKey finding:")
    print(f"  Cross-Attn + Static  (Phase 1) : {static_auroc:.4f}")
    print(f"  Cross-Attn + Dynamic (Phase 2) : {dynamic_auroc:.4f}")
    print(f"  Gain from dynamic over static  : {gain_str}")

    if gain > 0:
        print(f"\n  --> Dynamic reweighting outperforms static.")
        print(f"      Per-sample gating adds value beyond global alpha.")
    else:
        print(f"\n  --> Static matched or outperformed dynamic at 5 epochs.")
        print(f"      Run train_dynamic.py (15 epochs) for full dynamic results.")

    # ── Save ──────────────────────────────────────────────────────
    save_path = os.path.join(CONFIG["results_dir"], "ablation_results_full.txt")
    with open(save_path, "w") as f:
        f.write("FULL ABLATION STUDY -- Static + Dynamic Reweighting\n")
        f.write("=" * 74 + "\n")
        f.write(f"{'Model':<51} {'AUROC':>7} {'Acc':>7} {'F1':>7} {'ΔAUROC':>8}\n")
        f.write("-" * 74 + "\n")
        for r in all_results:
            delta     = r["auroc"] - baseline_auroc
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            f.write(f"{r['model']:<51}   {r['auroc']:>7.4f} {r['acc']:>7.4f} "
                    f"{r['f1']:>7.4f} {delta_str:>8}\n")
        f.write("=" * 74 + "\n")
        f.write(f"\nGain from dynamic over static: {gain_str}\n")

    print(f"\nSaved --> {save_path}")


if __name__ == "__main__":
    main()
