"""
train_dynamic_v3.py
===================
Thin orchestrator for AdaptiveFusionModelV3 — Dynamic Reweighting with all V3
improvements. All reusable logic lives in the shared modules:

    config.py         — defaults, env-var overrides, device selection
    dataset.py        — HatefulMemesDatasetV2 (+ input augmentation)
    losses.py         — FocalLoss, SupConLoss
    augmentation.py   — feature-level MixUp, R-Drop KL
    contrastive.py    — two-view SupCon wiring
    regularization.py — alpha entropy / diversity terms
    curriculum.py     — CLIP-similarity easy-first curriculum
    checkpointing.py  — best-AUROC + rolling periodic checkpoints
    train_utils.py    — train/eval loops, optimizer, threshold search

V3 training recipe (weights in CONFIG below)
--------------------------------------------
  MixUp · R-Drop · auxiliary per-modality heads · SupCon ·
  alpha entropy + diversity regularisation · curriculum learning ·
  input augmentation · attention dropout 0.2

Usage
-----
    python train_dynamic_v3.py

    # Colab/Kaggle: override paths and hyperparameters via env vars,
    # e.g. DATA_DIR, CHECKPOINT_DIR, NUM_EPOCHS — see config.py.
    # Set RESUME=true to continue from the rolling periodic checkpoint.
"""

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from checkpointing import CheckpointManager
from config        import apply_env_overrides, base_config, get_device
from curriculum    import build_curriculum_loaders
from dataset       import HatefulMemesDatasetV2
from losses        import FocalLoss, SupConLoss
from model         import AdaptiveFusionModelV3
from train_utils   import build_optimizer, evaluate, print_epoch_metrics, train_one_epoch


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    **base_config(),

    # ── V3-specific losses ────────────────────────────────────────────────────
    "mixup_alpha":      0.2,      # Beta(α, α) for feature MixUp; 0 = disabled
    "rdrop_weight":     0.5,      # symmetric KL between two stochastic passes
    "aux_weight":       0.3,      # auxiliary head focal losses
    "entropy_weight":   0.05,     # maximise alpha entropy to resist 0/1 gate collapse
    "diversity_weight": 0.05,     # encourage per-batch alpha spread
    "supcon_weight":    0.1,      # supervised contrastive loss weight
    "supcon_temp":      0.07,     # SupCon temperature

    # ── curriculum learning ───────────────────────────────────────────────────
    "curriculum":             True,   # set False to disable
    "curriculum_easy_thresh": 0.25,   # CLIP cos-sim floor for easy subset
    "curriculum_phase1_end":  3,      # epochs of easy-only training before full set

    # ── checkpointing ─────────────────────────────────────────────────────────
    "checkpoint_name": "best_model_dynamic_v3.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    apply_env_overrides(CONFIG)

    device = get_device()
    print(f"Device     : {device}")
    print(f"CLIP model : {CONFIG['clip_model']}")

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

    # ── data loaders (with optional curriculum) ───────────────────────────────
    loader_kwargs = dict(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )

    if CONFIG.get("curriculum", False):
        phase1_loader, full_loader = build_curriculum_loaders(
            train_ds, CONFIG["clip_model"], device, CONFIG
        )
        print(f"[Curriculum] Phase-1 ends after epoch {CONFIG['curriculum_phase1_end']}")
    else:
        full_loader   = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
        phase1_loader = full_loader   # unused

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── model ─────────────────────────────────────────────────────────────────
    model = AdaptiveFusionModelV3(
        clip_model_name=CONFIG["clip_model"],
        unfreeze_layers=CONFIG["unfreeze_layers"],
        attn_dropout=CONFIG["attn_dropout"],
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    # ── losses ────────────────────────────────────────────────────────────────
    criterion   = FocalLoss(
        alpha=CONFIG["focal_alpha"],
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["label_smoothing"],
    )
    supcon_loss = SupConLoss(
        temperature=CONFIG["supcon_temp"],
        base_temperature=CONFIG["supcon_temp"],
    )

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer     = build_optimizer(model, CONFIG)
    phase1_epochs = (
        min(CONFIG["curriculum_phase1_end"], CONFIG["num_epochs"])
        if CONFIG.get("curriculum", False)
        else 0
    )
    full_epochs = CONFIG["num_epochs"] - phase1_epochs
    total_steps = (
        phase1_epochs * (len(phase1_loader) // CONFIG["grad_accum"])
        + full_epochs * (len(full_loader) // CONFIG["grad_accum"])
    )
    total_steps  = max(1, total_steps)
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

        # curriculum: phase 1 = easy subset, phase 2 = full dataset
        train_loader = (
            phase1_loader
            if CONFIG.get("curriculum") and epoch <= CONFIG["curriculum_phase1_end"]
            else full_loader
        )

        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, CONFIG,
            supcon_loss=supcon_loss,
            is_dynamic=True,
        )

        metrics = evaluate(model, val_loader, criterion, device, is_dynamic=True)

        print_epoch_metrics(
            f"EPOCH {epoch}/{CONFIG['num_epochs']} — Dynamic V3",
            train_loss, train_auroc, metrics,
        )

        # ── checkpoint on best AUROC ──────────────────────────────────────────
        is_best = ckpt_mgr.save_best_if_improved(
            metrics["auroc"], epoch, model, optimizer,
            extra={
                "val_acc":       metrics["accuracy"],
                "val_acc_opt":   metrics["accuracy_opt"],
                "opt_threshold": metrics["opt_thresh"],
                "alpha_mean":    metrics["alpha_mean"],
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
    print("Training complete — Dynamic V3")
    print(f"Best Val AUROC  : {ckpt_mgr.best_metric:.4f}")
    print(f"Best Val Acc@opt: {best_acc_opt:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
