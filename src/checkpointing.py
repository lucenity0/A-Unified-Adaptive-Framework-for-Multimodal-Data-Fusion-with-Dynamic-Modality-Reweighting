"""
checkpointing.py
================
Checkpoint management for the V3 training scripts.

Two independent mechanisms:

  1. Best-metric checkpoint (existing behaviour, moved here unchanged):
     saved whenever validation AUROC improves. The payload layout is kept
     identical to what ensemble.py expects (model_state, optimizer_state,
     val_auroc, config, ...).

  2. Periodic checkpoint (new): a *rolling* file written every
     `periodic_every` epochs regardless of AUROC, containing full resume
     state (optimizer, scheduler, patience counter, best metric so far).
     A Colab/Kaggle disconnect mid-plateau then costs at most
     `periodic_every` epochs instead of the whole run. Rolling (single file,
     overwritten) because each checkpoint is ~2.5 GB.
"""

import os

import torch


class CheckpointManager:
    """
    Parameters
    ----------
    checkpoint_dir : directory for all checkpoint files (created if missing)
    best_name      : filename of the best-metric checkpoint, e.g.
                     "best_model_dynamic_v3.pt"
    periodic_every : save a rolling resume checkpoint every N epochs;
                     0 disables periodic saving
    """

    def __init__(self, checkpoint_dir: str, best_name: str,
                 periodic_every: int = 0):
        self.checkpoint_dir = checkpoint_dir
        self.best_name      = best_name
        self.periodic_every = periodic_every
        self.best_metric    = float("-inf")

        stem, ext = os.path.splitext(best_name)
        self.periodic_name = f"{stem}_last{ext}"

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── best checkpoint ───────────────────────────────────────────────────────
    def save_best_if_improved(self, metric: float, epoch: int, model,
                              optimizer, extra: dict = None) -> bool:
        """
        Save the best-metric checkpoint if `metric` beats the best seen so far.
        `extra` carries script-specific payload fields (val_acc, opt_threshold,
        alpha_mean / static_alpha, config, ...) exactly as before.

        Returns True if a new best was saved.
        """
        if metric <= self.best_metric:
            return False

        self.best_metric = metric
        payload = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_auroc":       metric,
        }
        if extra:
            payload.update(extra)

        path = os.path.join(self.checkpoint_dir, self.best_name)
        torch.save(payload, path)
        return True

    # ── periodic (rolling) checkpoint ─────────────────────────────────────────
    def save_periodic(self, epoch: int, model, optimizer, scheduler=None,
                      patience_ctr: int = 0, config: dict = None,
                      extra: dict = None) -> bool:
        """
        Every `periodic_every` epochs, overwrite the rolling resume checkpoint
        with full training state. `extra` carries trainer-local values that
        must survive a resume (e.g. best_acc_opt). Returns True if a save
        happened.
        """
        if self.periodic_every <= 0 or epoch % self.periodic_every != 0:
            return False

        payload = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_metric":     self.best_metric,
            "patience_ctr":    patience_ctr,
            "config":          config,
            "extra":           extra or {},
        }
        path = os.path.join(self.checkpoint_dir, self.periodic_name)
        torch.save(payload, path)
        print(f"💾  Periodic checkpoint saved (epoch {epoch}) → {path}")
        return True

    # ── resume ────────────────────────────────────────────────────────────────
    def load_resume(self, model, optimizer=None, scheduler=None,
                    map_location="cpu"):
        """
        Restore training state from the rolling periodic checkpoint, if one
        exists. Returns (start_epoch, patience_ctr, extra) — start_epoch is
        the next epoch to run (1 if nothing to resume), extra is the dict
        passed to save_periodic (empty if none).
        """
        path = os.path.join(self.checkpoint_dir, self.periodic_name)
        if not os.path.exists(path):
            return 1, 0, {}

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])

        self.best_metric = ckpt.get("best_metric", float("-inf"))
        start_epoch  = ckpt["epoch"] + 1
        patience_ctr = ckpt.get("patience_ctr", 0)
        print(f"⏯  Resumed from {path} — continuing at epoch {start_epoch} "
              f"(best AUROC so far: {self.best_metric:.4f})")
        return start_epoch, patience_ctr, ckpt.get("extra", {})
