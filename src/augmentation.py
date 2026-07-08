"""
augmentation.py
===============
Training-time regularisation applied at the *feature* level: MixUp and R-Drop.
(Input-level image/text augmentation lives in dataset.py.)

Both are used only by the dynamic V3 pipeline — they require the
extract_features() / fuse_and_classify() split of AdaptiveFusionModelV3.
"""

import numpy as np
import torch


def feature_mixup(
    txt_feat: torch.Tensor,
    img_feat: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
):
    """
    Feature-level MixUp (Zhang et al., 2018) applied jointly to the text and
    image feature vectors with a single λ ~ Beta(alpha, alpha) per batch.

    Returns
    -------
    (txt_mixed, img_mixed, labels_a, labels_b, lam)

    With alpha <= 0 MixUp is disabled: features pass through unchanged and
    lam = 1.0, so the blended loss  lam·L(y_a) + (1-lam)·L(y_b)  reduces to
    the plain loss.
    """
    if alpha <= 0.0:
        return txt_feat, img_feat, labels, labels, 1.0

    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(txt_feat.size(0), device=txt_feat.device)

    txt_mixed = lam * txt_feat + (1.0 - lam) * txt_feat[perm]
    img_mixed = lam * img_feat + (1.0 - lam) * img_feat[perm]

    return txt_mixed, img_mixed, labels, labels[perm], lam


def rdrop_kl(logit1: torch.Tensor, logit2: torch.Tensor) -> torch.Tensor:
    """
    R-Drop (Wu et al., 2021): symmetric KL divergence between the Bernoulli
    output distributions of two stochastic forward passes over the same input.

    Parameters
    ----------
    logit1, logit2 : (B,) raw logits from two dropout-independent passes.

    Returns
    -------
    scalar loss tensor
    """
    p1 = torch.sigmoid(logit1).clamp(1e-7, 1 - 1e-7)
    p2 = torch.sigmoid(logit2).clamp(1e-7, 1 - 1e-7)
    kl_12 = p1 * (p1 / p2).log() + (1 - p1) * ((1 - p1) / (1 - p2)).log()
    kl_21 = p2 * (p2 / p1).log() + (1 - p2) * ((1 - p2) / (1 - p1)).log()
    return (kl_12 + kl_21).mean() / 2.0
