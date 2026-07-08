"""
train_static_v3.py
==================
Thin orchestrator for StaticFusionModelV2 — Static Reweighting V3 baseline.

Keeps the training setup identical to train_dynamic_v3.py for a fair
comparison, with the sole architectural difference being a fixed α = 0.5
fusion weight instead of the learned uncertainty-aware gating network.
Shares config.py / train_utils.py / checkpointing.py with the dynamic script.

Omissions vs dynamic V3 (intentional — static model has no aux heads)
----------------------------------------------------------------------
  • No MixUp          (StaticFusionModelV2 has no extract_features() split)
  • No R-Drop         (no fuse_and_classify() — single unified forward)
  • No auxiliary head losses (no img_aux_head / txt_aux_head)
  • No alpha entropy/diversity reg (alpha is constant)
  • No SupConLoss     (disabled by supcon_weight = 0.0)
  • No curriculum     (clean baseline)

Everything else matches dynamic V3:
  focal loss  |  label smoothing  |  grad accum  |  cosine-warmup LR
  dataset augmentation  |  differential LR  |  same shared CONFIG values

Usage
-----
    python train_static_v3.py

    # Colab/Kaggle: override paths and hyperparameters via env vars,
    # e.g. DATA_DIR, CHECKPOINT_DIR, NUM_EPOCHS — see config.py.
    # Set RESUME=true to continue from the rolling periodic checkpoint.
"""

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from checkpointing import CheckpointManager
from config        import apply_env_overrides, base_config, get_device
from dataset       import HatefulMemesDatasetV2
from losses        import FocalLoss
from model         import StaticFusionModelV2
from train_utils   import build_optimizer, evaluate, print_epoch_metrics, train_one_epoch


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    **base_config(),

    # ── static baseline ───────────────────────────────────────────────────────
    "static_alpha":     0.5,      # fixed fusion weight

    # ── V3 losses — all disabled for static baseline ──────────────────────────
    "mixup_alpha":      0.0,      # disabled — no split forward pass
    "rdrop_weight":     0.0,      # disabled
    "aux_weight":       0.0,      # disabled — no aux heads
    "entropy_weight":   0.0,      # disabled — alpha is constant
    "diversity_weight": 0.0,      # disabled — no gating network
    "supcon_weight":    0.0,      # disabled for clean ablation

    # ── checkpointing ─────────────────────────────────────────────────────────
    "checkpoint_name": "best_model_static_v3.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    apply_env_overrides(CONFIG)

    device = get_device()
    print(f"Device      : {device}")
    print(f"CLIP model  : {CONFIG['clip_model']}")
    print(f"Static α    : {CONFIG['static_alpha']}")

    # ── processor + datasets ──────────────────────────────────────────────────
    processor = CLIPProcessor.from_pretrained(CONFIG["clip_model"])

    train_ds = HatefulMemesDatasetV2(
        CONFIG["train_parquet"], processor, augment=True,
        max_samples=CONFIG.get("max_samples", 0),
    )
    val_ds   = HatefulMemesDatasetV2(
        CONFIG["val_parquet"],   processor, augment=False,
        max_samples=CONFIG.get("max_samples", 0),
    )

    loader_kwargs = dict(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── model ─────────────────────────────────────────────────────────────────
    model = StaticFusionModelV2(
        clip_model_name=CONFIG["clip_model"],
        unfreeze_layers=CONFIG["unfreeze_layers"],
        static_alpha=CONFIG["static_alpha"],
        attn_dropout=CONFIG["attn_dropout"],
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = FocalLoss(
        alpha=CONFIG["focal_alpha"],
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["label_smoothing"],
    )

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer    = build_optimizer(model, CONFIG)
    total_steps  = max(1, CONFIG["num_epochs"] * len(train_loader) // CONFIG["grad_accum"])
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── checkpointing (best-AUROC + rolling periodic, optional resume) ────────
    ckpt_mgr = CheckpointManager(
        CONFIG["checkpoint_dir"],
        CONFIG["checkpoint_name"],
        periodic_every=CONFIG.get("checkpoint_every", 0),
    )
    start_epoch, patience_ctr, best_acc_opt = 1, 0, 0.0
    if CONFIG.get("resume", False):
        start_epoch, patience_ctr, resume_extra = ckpt_mgr.load_resume(
            model, optimizer, scheduler, map_location=device
        )
        best_acc_opt = resume_extra.get("best_acc_opt", 0.0)

    # ── training loop ─────────────────────────────────────────────────────────

    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):

        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, CONFIG,
            supcon_loss=None,
            is_dynamic=False,   # ← static model path
        )

        metrics = evaluate(model, val_loader, criterion, device, is_dynamic=False)

        print_epoch_metrics(
            f"EPOCH {epoch}/{CONFIG['num_epochs']} — Static V3  (α={CONFIG['static_alpha']})",
            train_loss, train_auroc, metrics,
        )

        # ── checkpoint on best AUROC ──────────────────────────────────────────
        is_best = ckpt_mgr.save_best_if_improved(
            metrics["auroc"], epoch, model, optimizer,
            extra={
                "val_acc":       metrics["accuracy"],
                "val_acc_opt":   metrics["accuracy_opt"],
                "opt_threshold": metrics["opt_thresh"],
                "static_alpha":  CONFIG["static_alpha"],
                "config":        CONFIG,
            },
        )

        if is_best:
            best_acc_opt = metrics["accuracy_opt"]
            patience_ctr = 0
            print(f"✅  Checkpoint saved — AUROC: {ckpt_mgr.best_metric:.4f} | "
                  f"Acc@opt: {best_acc_opt:.4f}")
        else:
            patience_ctr += 1
            print(f"No improvement — patience {patience_ctr}/{CONFIG['patience']}")

        # ── periodic (rolling) checkpoint, independent of AUROC ───────────────
        ckpt_mgr.save_periodic(
            epoch, model, optimizer, scheduler,
            patience_ctr=patience_ctr, config=CONFIG,
            extra={"best_acc_opt": best_acc_opt},
        )

        if patience_ctr >= CONFIG["patience"]:
            print("Early stopping triggered.")
            break

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Training complete — Static V3")
    print(f"Best Val AUROC  : {ckpt_mgr.best_metric:.4f}")
    print(f"Best Val Acc@opt: {best_acc_opt:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
