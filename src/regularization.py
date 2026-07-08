"""
regularization.py
=================
Regularisation terms on the dynamic gating output α ∈ (0,1).

Both terms directly target the alpha-collapse observed in V2 runs, where the
per-sample fusion weight's std dev shrank from 0.035 to 0.011 over 10 epochs:

  • alpha_entropy_loss   — keeps individual α values away from saturating at
                           0/1 by maximising per-sample Bernoulli entropy.
  • alpha_diversity_loss — keeps the *batch* of α values spread out by
                           maximising batch-level variance, so the gate stays
                           content-aware instead of converging to a constant.

Both return terms to be *added* to the total loss (they are already negated
where maximisation is intended).
"""

import torch


def alpha_entropy_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    Negative mean Bernoulli entropy of α. Minimising this maximises H(α),
    resisting 0/1 gate saturation.
    """
    a = alpha.clamp(1e-7, 1 - 1e-7)
    return (a * a.log() + (1 - a) * (1 - a).log()).mean()


def alpha_diversity_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    Negative batch variance of α. Minimising this maximises the spread of
    fusion weights within a batch, counteracting collapse to a single value.
    """
    return -torch.var(alpha)
