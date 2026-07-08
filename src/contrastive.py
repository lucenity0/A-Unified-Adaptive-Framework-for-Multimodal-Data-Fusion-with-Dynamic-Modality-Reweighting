"""
contrastive.py
==============
SupCon wiring for the dynamic V3 pipeline.

The SupConLoss implementation itself lives in losses.py; this module holds
the view construction: each sample contributes two views of the same
underlying meme — its (L2-normalised) text feature and image feature — so
pairs from the same class attract across modalities.
"""

import torch


def two_view_supcon(
    txt_feat: torch.Tensor,
    img_feat: torch.Tensor,
    labels: torch.Tensor,
    supcon_loss,
) -> torch.Tensor:
    """
    Compute the supervised contrastive loss over the two-view stack
    [text, image] of clean (pre-MixUp) features.

    Parameters
    ----------
    txt_feat, img_feat : (B, D) L2-normalised feature vectors
    labels             : (B,)  binary float labels
    supcon_loss        : losses.SupConLoss instance (or None to skip)

    Returns
    -------
    scalar loss tensor (0.0 when supcon_loss is None)
    """
    if supcon_loss is None:
        return torch.tensor(0.0, device=txt_feat.device)

    two_view = torch.stack([txt_feat, img_feat], dim=1)   # (B, 2, D)
    return supcon_loss(two_view, labels.long())
