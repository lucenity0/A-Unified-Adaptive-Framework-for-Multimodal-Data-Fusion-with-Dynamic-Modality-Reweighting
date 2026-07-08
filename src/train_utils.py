"""
train_utils.py
==============
Shared training / evaluation utilities for V3 dynamic and static pipelines.

Public API
----------
find_optimal_threshold(labels, probs)
    → (threshold: float, accuracy: float)

build_optimizer(model, cfg)
    → AdamW with differential learning rates (CLIP vs fusion modules)

train_one_epoch(model, loader, optimizer, scheduler, criterion,
                device, cfg, supcon_loss=None, is_dynamic=True)
    → (avg_loss: float, auroc: float)

evaluate(model, loader, criterion, device, is_dynamic=True)
    → dict with loss / auroc / accuracy / f1 / opt_thresh / alpha stats

print_epoch_metrics(tag, train_loss, train_auroc, m)
    → formatted per-epoch console summary

V3 loss terms used during dynamic training (weights all come from cfg —
missing keys mean the term is disabled):
  • Feature-level MixUp      (cfg["mixup_alpha"])     — augmentation.feature_mixup
  • R-Drop                   (cfg["rdrop_weight"])    — augmentation.rdrop_kl
  • Auxiliary head losses    (cfg["aux_weight"])      — img_aux + txt_aux focal loss
  • Supervised contrastive   (cfg["supcon_weight"])   — contrastive.two_view_supcon
  • Alpha entropy reg.       (cfg["entropy_weight"])  — regularization.alpha_entropy_loss
  • Alpha diversity loss     (cfg["diversity_weight"])— regularization.alpha_diversity_loss
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from augmentation   import feature_mixup, rdrop_kl
from contrastive    import two_view_supcon
from regularization import alpha_diversity_loss, alpha_entropy_loss


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    n_steps: int = 100,
) -> tuple[float, float]:
    """
    Grid-search the decision threshold that maximises accuracy on a held-out
    set.

    Returns
    -------
    (best_threshold, best_accuracy)
    """
    best_thresh = 0.5
    best_acc    = 0.0

    for thresh in np.linspace(0.1, 0.9, n_steps):
        preds = (probs >= thresh).astype(int)
        acc   = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh

    return float(best_thresh), float(best_acc)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, cfg: dict):
    """AdamW with differential learning rates: lower LR for the fine-tuned
    CLIP layers, higher for the newly introduced fusion modules."""
    clip_params   = []
    fusion_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "clip" in name:
            clip_params.append(param)
        else:
            fusion_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": clip_params,   "lr": cfg["clip_lr"]},
            {"params": fusion_params, "lr": cfg["fusion_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    cfg: dict,
    supcon_loss=None,
    is_dynamic: bool = True,
):
    """
    Run one training epoch.

    Parameters
    ----------
    model        : AdaptiveFusionModelV3 (is_dynamic=True)
                   or StaticFusionModelV2 (is_dynamic=False)
    loader       : DataLoader
    optimizer    : AdamW
    scheduler    : cosine-warmup scheduler
    criterion    : FocalLoss instance
    device       : torch.device
    cfg          : config dict (see train_dynamic_v3.py for all keys)
    supcon_loss  : SupConLoss instance or None
    is_dynamic   : if False, skip MixUp / R-Drop / aux heads (static model)

    Returns
    -------
    (avg_train_loss, train_auroc)
    """
    model.train()

    grad_accum = cfg.get("grad_accum", 4)

    total_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["label"].to(device)

        # ── DYNAMIC path ─────────────────────────────────────────────────────
        if is_dynamic:
            loss, logit_for_auroc = _dynamic_step(
                model, input_ids, attention_mask, pixel_values, labels,
                criterion, supcon_loss, cfg,
            )

        # ── STATIC path ──────────────────────────────────────────────────────
        else:
            loss, logit_for_auroc = _static_step(
                model, input_ids, attention_mask, pixel_values, labels,
                criterion,
            )

        probs = torch.sigmoid(logit_for_auroc).detach().cpu().numpy()

        # ── gradient accumulation ─────────────────────────────────────────────
        (loss / grad_accum).backward()

        if (batch_idx + 1) % grad_accum == 0:
            _clipped_step(model, optimizer)
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    # flush remaining gradients
    if len(loader) % grad_accum != 0:
        _clipped_step(model, optimizer)
        optimizer.zero_grad()

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auroc = 0.0

    return total_loss / len(loader), auroc


def _clipped_step(model, optimizer, max_norm: float = 1.0):
    """Clip gradients and step — unless the gradient norm is non-finite.

    A rare bad batch (or an MPS/CUDA backward quirk in the CLIP encoder) can
    yield inf gradients; clip_grad_norm_ would then rescale them to NaN and a
    single optimizer.step() would corrupt the weights for the rest of the
    run. Skipping the step discards that one batch's update instead.
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        print("  [warn] non-finite gradient norm — skipping optimizer step")


# ── inner step helpers ────────────────────────────────────────────────────────

def _dynamic_step(
    model, input_ids, attention_mask, pixel_values, labels,
    criterion, supcon_loss, cfg: dict,
) -> tuple:
    """
    Full V3 dynamic training step:
      1. Extract features once   (expensive CLIP pass)
      2. SupCon loss on clean features
      3. Optional feature MixUp
      4. Two stochastic fuse_and_classify passes (R-Drop)
      5. Main focal loss  + R-Drop KL  + aux losses  + SupCon
         + entropy reg  + diversity loss

    All term weights come from cfg; a missing/zero weight disables the term.

    Returns
    -------
    (loss, logit1)   — logit1 reused for training AUROC logging (no extra pass)
    """
    mixup_alpha      = cfg.get("mixup_alpha",      0.0)
    rdrop_weight     = cfg.get("rdrop_weight",     0.0)
    aux_weight       = cfg.get("aux_weight",       0.0)
    supcon_weight    = cfg.get("supcon_weight",    0.0)
    entropy_weight   = cfg.get("entropy_weight",   0.0)
    diversity_weight = cfg.get("diversity_weight", 0.0)

    # ── 1. Feature extraction ─────────────────────────────────────────────────
    txt_feat, img_feat = model.extract_features(input_ids, attention_mask, pixel_values)

    # ── 2. SupCon loss on clean features ─────────────────────────────────────
    loss_supcon = (
        two_view_supcon(txt_feat, img_feat, labels, supcon_loss)
        if supcon_weight > 0.0
        else torch.tensor(0.0, device=txt_feat.device)
    )

    # ── 3. MixUp at feature level ─────────────────────────────────────────────
    txt_m, img_m, lab_a, lab_b, lam = feature_mixup(
        txt_feat, img_feat, labels, mixup_alpha
    )

    # ── 4. Two stochastic forward passes (R-Drop) ─────────────────────────────
    logit1, alpha1, img_aux1, txt_aux1 = model.fuse_and_classify(txt_m, img_m)
    logit2, alpha2, img_aux2, txt_aux2 = model.fuse_and_classify(txt_m, img_m)

    # ── 5a. Main focal loss (MixUp-blended) ───────────────────────────────────
    focal1 = lam * criterion(logit1, lab_a) + (1.0 - lam) * criterion(logit1, lab_b)
    focal2 = lam * criterion(logit2, lab_a) + (1.0 - lam) * criterion(logit2, lab_b)
    loss_main = (focal1 + focal2) / 2.0

    # ── 5b. R-Drop symmetric KL divergence ────────────────────────────────────
    loss_rdrop = rdrop_kl(logit1, logit2)

    # ── 5c. Auxiliary head losses ─────────────────────────────────────────────
    aux_img = lam * criterion(img_aux1, lab_a) + (1.0 - lam) * criterion(img_aux1, lab_b)
    aux_txt = lam * criterion(txt_aux1, lab_a) + (1.0 - lam) * criterion(txt_aux1, lab_b)
    loss_aux = (aux_img + aux_txt) / 2.0

    # ── 5d/5e. Alpha regularisation ───────────────────────────────────────────
    loss_entropy   = alpha_entropy_loss(alpha1)
    loss_diversity = alpha_diversity_loss(alpha1)

    # ── 5f. Total loss ────────────────────────────────────────────────────────
    loss = (
        loss_main
        + rdrop_weight     * loss_rdrop
        + aux_weight       * loss_aux
        + entropy_weight   * loss_entropy
        + diversity_weight * loss_diversity
        + supcon_weight    * loss_supcon
    )
    return loss, logit1.detach()


def _static_step(
    model, input_ids, attention_mask, pixel_values, labels, criterion
) -> tuple:
    """Simple forward pass for the static model (no aux heads, no MixUp).
    Returns (loss, logit) so the caller can log training AUROC without a
    second expensive forward pass."""
    logits, _ = model(input_ids, attention_mask, pixel_values)
    return criterion(logits, labels), logits.detach()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    loader,
    criterion,
    device,
    is_dynamic: bool = True,
) -> dict:
    """
    Full evaluation pass.

    Returns a dict with keys:
        loss, auroc, accuracy, accuracy_opt, f1, f1_opt,
        opt_thresh, alpha_mean, alpha_std, alpha_min, alpha_max
    """
    model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_alphas = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            labels         = batch["label"].to(device)

            if is_dynamic:
                logits, alpha, img_aux, txt_aux = model(
                    input_ids, attention_mask, pixel_values
                )
                # Evaluation loss: main head only
                loss = criterion(logits, labels)
            else:
                logits, alpha = model(input_ids, attention_mask, pixel_values)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_alphas.extend(alpha.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_alphas = np.array(all_alphas)

    # ── threshold search ──────────────────────────────────────────────────────
    opt_thresh, opt_acc = find_optimal_threshold(all_labels, all_preds)

    # ── metrics ───────────────────────────────────────────────────────────────
    bin_default = (all_preds >= 0.5).astype(int)
    bin_opt     = (all_preds >= opt_thresh).astype(int)

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auroc = 0.0

    return {
        "loss":         total_loss / len(loader),
        "auroc":        auroc,
        "accuracy":     accuracy_score(all_labels, bin_default),
        "accuracy_opt": accuracy_score(all_labels, bin_opt),
        "f1":           f1_score(all_labels, bin_default, average="macro"),
        "f1_opt":       f1_score(all_labels, bin_opt,     average="macro"),
        "opt_thresh":   opt_thresh,
        "alpha_mean":   float(all_alphas.mean()),
        "alpha_std":    float(all_alphas.std()),
        "alpha_min":    float(all_alphas.min()),
        "alpha_max":    float(all_alphas.max()),
    }


def collect_predictions(model, loader, device, is_dynamic: bool = True):
    """
    Run a full no-grad pass and return (probs, labels) as numpy arrays.
    Used by significance.py (and any post-hoc analysis) to get paired
    per-sample predictions on the same, un-shuffled loader.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)

            if is_dynamic:
                logits, *_ = model(input_ids, attention_mask, pixel_values)
            else:
                logits, _  = model(input_ids, attention_mask, pixel_values)

            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    return np.array(all_probs), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def print_epoch_metrics(tag: str, train_loss: float, train_auroc: float,
                        m: dict):
    """Per-epoch console summary shared by both training scripts."""
    print(f"\n{'='*65}")
    print(tag)
    print(f"{'='*65}")
    print(f"Train  ▸  Loss: {train_loss:.4f}  |  AUROC: {train_auroc:.4f}")
    print(f"Val    ▸  Loss: {m['loss']:.4f}  |  AUROC: {m['auroc']:.4f}")
    print(f"Val    ▸  Acc@0.5: {m['accuracy']:.4f}  |  "
          f"Acc@{m['opt_thresh']:.2f}: {m['accuracy_opt']:.4f}")
    print(f"Val    ▸  F1@0.5: {m['f1']:.4f}  |  F1@opt: {m['f1_opt']:.4f}")
    print(f"Alpha  ▸  Mean: {m['alpha_mean']:.3f}  |  Std: {m['alpha_std']:.3f}  |  "
          f"Range: [{m['alpha_min']:.3f}, {m['alpha_max']:.3f}]")
