"""
train_dynamic.py
================
Trains ONLY the Dynamic Reweighting model (Phase 2).
Saves best checkpoint to ../checkpoints/best_model_dynamic.pt

Phase 2 additions over Phase 1:
    - DynamicGatingNetwork: per-sample alpha computed from input
    - Entropy regularization: forces alpha to be diverse (not collapse to 0.5)

Run:
    python3 train_dynamic.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from dataset import get_dataloaders
from model import AdaptiveFusionModel


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "train_parquet"  : "../Data/train-00000-of-00001-6587b3a58d350036.parquet",
    "val_parquet"    : "../Data/validation-00000-of-00001-1508d9e5032c2c1f.parquet",
    "batch_size"     : 16,
    "num_epochs"     : 15,
    "freeze_clip"    : True,
    "clip_lr"        : 1e-5,
    "fusion_lr"      : 3e-4,
    "weight_decay"   : 0.05,
    "warmup_ratio"   : 0.1,
    "pos_weight"     : 1.5,
    "entropy_weight" : 0.01,        # forces dynamic alpha to be diverse
    "checkpoint_dir" : "../checkpoints",
    "checkpoint_name": "best_model_dynamic.pt",
    "patience"       : 5
}


# ─────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values   = batch['pixel_values'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        logits, alpha = model(input_ids, attention_mask, pixel_values)

        # Classification loss
        loss = criterion(logits, labels)

        # Entropy regularization -- forces alpha away from 0.5
        # Without this, dynamic gating collapses to ~0.5 for all samples
        eps = 1e-8
        alpha_entropy = -(alpha * torch.log(alpha + eps) +
                          (1 - alpha) * torch.log(1 - alpha + eps))
        entropy_loss  = -alpha_entropy.mean()

        # Combined loss
        loss = loss + CONFIG["entropy_weight"] * entropy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    auroc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auroc


# ─────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_alphas = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values   = batch['pixel_values'].to(device)
            labels         = batch['label'].to(device)

            logits, alpha = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_alphas.extend(alpha.mean(dim=-1).cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_alphas = np.array(all_alphas)
    bin_preds  = (all_preds >= 0.5).astype(int)

    auroc    = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, bin_preds)
    f1       = f1_score(all_labels, bin_preds, average='macro')

    return total_loss / len(loader), auroc, accuracy, f1, all_alphas


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print("Phase 2 -- Dynamic Reweighting Training")
    print("Entropy regularization ON (weight=0.01)")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        CONFIG["train_parquet"],
        CONFIG["val_parquet"],
        batch_size=CONFIG["batch_size"]
    )

    # ── Model ─────────────────────────────────────────────────────
    model = AdaptiveFusionModel(freeze_clip=CONFIG["freeze_clip"]).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,}")

    # ── Loss ──────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(device)
    )

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {'params': model.clip.parameters(),       'lr': CONFIG["clip_lr"]},
        {'params': model.cross_attn.parameters(), 'lr': CONFIG["fusion_lr"]},
        {'params': model.gating.parameters(),     'lr': CONFIG["fusion_lr"]},
        {'params': model.classifier.parameters(), 'lr': CONFIG["fusion_lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    # ── Scheduler ─────────────────────────────────────────────────
    total_steps  = CONFIG["num_epochs"] * len(train_loader)
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Training Loop ─────────────────────────────────────────────
    best_auroc   = 0.0
    patience_ctr = 0

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'='*52}")
        print(f"EPOCH {epoch+1}/{CONFIG['num_epochs']} -- Dynamic Reweighting")
        print(f"{'='*52}")

        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auroc, val_acc, val_f1, alphas = evaluate(
            model, val_loader, criterion, device
        )

        print(f"\nTrain --> Loss: {train_loss:.4f} | AUROC: {train_auroc:.4f}")
        print(f"Val   --> Loss: {val_loss:.4f} | AUROC: {val_auroc:.4f} | "
              f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"Alpha --> Mean: {alphas.mean():.3f} | Std: {alphas.std():.3f} | "
              f"Min: {alphas.min():.3f} | Max: {alphas.max():.3f}")

        # Watch Std -- if still below 0.05 after epoch 3,
        # increase entropy_weight to 0.05 in CONFIG
        if epoch >= 2 and alphas.std() < 0.05:
            print("  WARNING: Alpha Std still low -- consider increasing entropy_weight to 0.05")

        if val_auroc > best_auroc:
            best_auroc   = val_auroc
            patience_ctr = 0
            path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_name"])
            torch.save({
                'epoch'          : epoch + 1,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_auroc'      : val_auroc,
                'val_acc'        : val_acc,
                'val_f1'         : val_f1,
                'alpha_mean'     : float(alphas.mean()),
                'alpha_std'      : float(alphas.std()),
            }, path)
            print(f"Saved best model --> {CONFIG['checkpoint_name']} | AUROC: {best_auroc:.4f}")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{CONFIG['patience']}")
            if patience_ctr >= CONFIG["patience"]:
                print("Early stopping triggered.")
                break

    print(f"\nPhase 2 training complete.")
    print(f"Best Val AUROC : {best_auroc:.4f}")
    print(f"\nNext: run python3 compare_phases.py to see Phase 1 vs Phase 2 results")


if __name__ == "__main__":
    main()
