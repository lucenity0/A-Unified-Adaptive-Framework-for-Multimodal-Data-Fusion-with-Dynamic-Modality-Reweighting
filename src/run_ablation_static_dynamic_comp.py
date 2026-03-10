"""
run_ablation.py
===============
Trains Static and Dynamic Reweighting models and prints comparison table.

Variants:
    1. Cross-Attn + Static Reweighting  (Phase 1)
    2. Cross-Attn + Dynamic Reweighting (Phase 2)

Run:
    python3 run_ablation.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from dataset import get_dataloaders
from model import AdaptiveFusionModel, StaticGatingNetwork, DynamicGatingNetwork


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "train_parquet" : "../Data/train-00000-of-00001-6587b3a58d350036.parquet",
    "val_parquet"   : "../Data/validation-00000-of-00001-1508d9e5032c2c1f.parquet",
    "batch_size"    : 16,
    "num_epochs"    : 10,
    "freeze_clip"   : True,
    "fusion_lr"     : 3e-4,
    "clip_lr"       : 1e-5,
    "pos_weight"    : 1.5,
    "entropy_weight": 0.01,   # dynamic model only
    "results_dir"   : "../results"
}

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────────────────────────
# TRAIN + EVAL
# ─────────────────────────────────────────────────────────────────
def train_and_evaluate(model, model_name, train_loader, val_loader, use_entropy=False):

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    print(f"{'='*55}")

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(DEVICE)
    )

    clip_params  = list(model.clip.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('clip') and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': clip_params,  'lr': CONFIG["clip_lr"]},
        {'params': other_params, 'lr': CONFIG["fusion_lr"]},
    ], weight_decay=0.05)

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

            if use_entropy:
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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values   = batch['pixel_values'].to(DEVICE)
                labels         = batch['label']

                logits, _ = model(input_ids, attention_mask, pixel_values)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        bin_preds  = (all_preds >= 0.5).astype(int)

        auroc = roc_auc_score(all_labels, all_preds)
        acc   = accuracy_score(all_labels, bin_preds)
        f1    = f1_score(all_labels, bin_preds, average='macro')

        print(f"  Epoch {epoch+1}/{CONFIG['num_epochs']} → "
              f"AUROC: {auroc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_acc   = acc
            best_f1    = f1

    return {"model": model_name, "auroc": best_auroc, "acc": best_acc, "f1": best_f1}


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_loader, val_loader = get_dataloaders(
        CONFIG["train_parquet"],
        CONFIG["val_parquet"],
        batch_size=CONFIG["batch_size"]
    )

    results = []

    # ── Model 1: Static Reweighting ───────────────────────────────
    static_model        = AdaptiveFusionModel(freeze_clip=CONFIG["freeze_clip"])
    static_model.gating = StaticGatingNetwork()
    results.append(train_and_evaluate(
        static_model,
        "Cross-Attn + Static Reweighting (Phase 1)",
        train_loader, val_loader,
        use_entropy=False
    ))

    # ── Model 2: Dynamic Reweighting ─────────────────────────────
    dynamic_model = AdaptiveFusionModel(freeze_clip=CONFIG["freeze_clip"])
    results.append(train_and_evaluate(
        dynamic_model,
        "Cross-Attn + Dynamic Reweighting (Phase 2)",
        train_loader, val_loader,
        use_entropy=True
    ))

    # ── Print Results Table ───────────────────────────────────────
    static_auroc  = results[0]["auroc"]
    dynamic_auroc = results[1]["auroc"]
    gain          = dynamic_auroc - static_auroc
    gain_str      = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"

    print("\n")
    print("=" * 68)
    print(f"{'STATIC vs DYNAMIC REWEIGHTING -- ABLATION':^68}")
    print("=" * 68)
    print(f"{'Model':<44} {'AUROC':>7} {'Acc':>7} {'F1':>7}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<44} {r['auroc']:>7.4f} {r['acc']:>7.4f} {r['f1']:>7.4f}")
    print("-" * 68)
    print(f"{'Gain from Dynamic over Static':<44} {gain_str:>7}")
    print("=" * 68)

    # ── Save ──────────────────────────────────────────────────────
    save_path = os.path.join(CONFIG["results_dir"], "ablation_static_vs_dynamic.txt")
    with open(save_path, "w") as f:
        f.write("STATIC vs DYNAMIC REWEIGHTING -- ABLATION\n")
        f.write("=" * 68 + "\n")
        f.write(f"{'Model':<44} {'AUROC':>7} {'Acc':>7} {'F1':>7}\n")
        f.write("-" * 68 + "\n")
        for r in results:
            f.write(f"{r['model']:<44} {r['auroc']:>7.4f} {r['acc']:>7.4f} {r['f1']:>7.4f}\n")
        f.write("-" * 68 + "\n")
        f.write(f"Gain from Dynamic over Static: {gain_str}\n")
        f.write("=" * 68 + "\n")

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
