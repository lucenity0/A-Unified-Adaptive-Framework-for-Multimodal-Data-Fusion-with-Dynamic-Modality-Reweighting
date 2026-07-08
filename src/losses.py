"""
losses.py
=========
Loss functions used by V3 training pipelines.

  FocalLoss   — class-balanced focal loss with optional label smoothing.
                Identical to the version in model.py (kept here for clean
                imports without pulling in the heavy model file).

  SupConLoss  — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
                Supports two calling conventions:

                  (a) Two-view:
                        features : (B, 2, D)   — e.g. stack([txt, img], dim=1)
                        labels   : (B,)
                      Each sample has an image view and a text view; pairs
                      from the same class are positives.

                  (b) Flat:
                        features : (B, D)
                        labels   : (B,)
                      Standard single-view contrastive loss; positives are
                      other samples in the batch with the same label.

                In both cases features should be L2-normalised before
                being passed in (or set normalize=True).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss with optional label smoothing.

    Args
    ----
    alpha           : weight for the positive class (default 0.75).
    gamma           : focusing exponent (default 2.0).
    label_smoothing : soft-label epsilon (default 0.0).
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B,)  raw (un-sigmoided) scores
        targets : (B,)  binary float labels in {0, 1}
        """
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs   = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss    = alpha_t * (1.0 - p_t) ** self.gamma * ce_loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISED CONTRASTIVE LOSS
# ─────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — Khosla et al., NeurIPS 2020.
    https://arxiv.org/abs/2004.11362

    Args
    ----
    temperature      : logit scale τ (default 0.07).
    base_temperature : reference temperature for loss scaling (default 0.07).
    normalize        : L2-normalise features before computing similarities
                       (default True).  Set False if features are already
                       normalised.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature
        self.normalize        = normalize

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, D)  or  (B, n_views, D)
        labels   : (B,)   integer class indices

        Returns
        -------
        scalar loss tensor
        """
        device = features.device

        # ── normalise ─────────────────────────────────────────────────────────
        if self.normalize:
            if features.dim() == 3:                          # (B, V, D)
                features = F.normalize(features, dim=-1)
            else:                                            # (B, D)
                features = F.normalize(features, dim=-1)

        # ── flatten to (N, D) ─────────────────────────────────────────────────
        if features.dim() == 3:
            B, V, D = features.shape
            # Repeat labels V times: [0,1,2,...] → [0,0,1,1,2,2,...]
            labels   = labels.unsqueeze(1).expand(B, V).reshape(-1)   # (N,)
            features = features.reshape(B * V, D)                     # (N, D)
        else:
            B = features.size(0)

        N = features.size(0)

        # ── similarity matrix ─────────────────────────────────────────────────
        # (N, N)  —  each entry is cos-sim(i, j) / τ
        sim = torch.matmul(features, features.T) / self.temperature   # (N, N)

        # ── masks ─────────────────────────────────────────────────────────────
        labels    = labels.view(-1, 1)                                 # (N, 1)
        pos_mask  = (labels == labels.T).float().to(device)            # (N, N)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        not_self  = (~self_mask).float()
        pos_mask  = pos_mask * not_self   # exclude self-pair from positives

        # ── numerically stable log-sum-exp ────────────────────────────────────
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()                            # stability

        exp_sim = torch.exp(sim) * not_self   # exclude self from denominator

        # ── log-probability ───────────────────────────────────────────────────
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # ── mean log-prob over positive pairs ─────────────────────────────────
        n_pos = pos_mask.sum(dim=1)                                    # (N,)
        # Samples with no positive pair in the batch → skip (set loss to 0)
        valid        = n_pos > 0
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = loss[valid].mean() if valid.any() else loss.mean() * 0.0
        return loss
