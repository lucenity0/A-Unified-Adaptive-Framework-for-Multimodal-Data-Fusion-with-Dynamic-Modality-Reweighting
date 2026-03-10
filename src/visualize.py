"""
visualize.py
============
Visualize dynamic modality weights (alpha) — your key research contribution.
Run AFTER training to generate plots for your report.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import AdaptiveFusionModel


def plot_alpha_distribution(model, val_loader, device, save_path="alpha_distribution.png"):
    """
    Shows how alpha (image weight) is distributed across hateful vs not-hateful memes.

    Key insight:
    - If hateful memes cluster at high alpha → image is the hateful signal
    - If hateful memes cluster at low alpha  → text is the hateful signal
    - Spread distribution                   → both modalities matter (multimodal!)
    """
    model.eval()
    all_alphas, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values   = batch['pixel_values'].to(device)
            labels         = batch['label']

            _, alpha = model(input_ids, attention_mask, pixel_values)

            # Mean alpha across 512 dimensions → one scalar per sample
            alpha_mean = alpha.mean(dim=-1).cpu().numpy()
            all_alphas.extend(alpha_mean)
            all_labels.extend(labels.numpy())

    all_alphas = np.array(all_alphas)
    all_labels = np.array(all_labels)

    hateful_alphas     = all_alphas[all_labels == 1]
    not_hateful_alphas = all_alphas[all_labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1: Histogram ────────────────────────────────────────
    axes[0].hist(not_hateful_alphas, bins=30, alpha=0.7,
                 label=f'Not Hateful (n={len(not_hateful_alphas)})', color='steelblue')
    axes[0].hist(hateful_alphas,     bins=30, alpha=0.7,
                 label=f'Hateful (n={len(hateful_alphas)})',     color='tomato')
    axes[0].set_xlabel("Mean Alpha\n← Text dominant        Image dominant →", fontsize=11)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dynamic Modality Weight Distribution")
    axes[0].legend()
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=1, label='Equal weight')

    # ── Plot 2: Boxplot ──────────────────────────────────────────
    axes[1].boxplot(
        [not_hateful_alphas, hateful_alphas],
        labels=['Not Hateful', 'Hateful'],
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.7),
    )
    axes[1].set_ylabel("Mean Alpha (Image Weight)")
    axes[1].set_title("Alpha Distribution: Hateful vs Not Hateful")
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")
    print(f"\nSummary:")
    print(f"  Hateful     — Mean alpha: {hateful_alphas.mean():.3f} ± {hateful_alphas.std():.3f}")
    print(f"  Not Hateful — Mean alpha: {not_hateful_alphas.mean():.3f} ± {not_hateful_alphas.std():.3f}")


if __name__ == "__main__":
    # FOR M2 MAC
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    _, val_loader = get_dataloaders(
        "train-00000-of-0000....parquet",
        "validation-00000-of-000....parquet",
        batch_size=32
    )

    model = AdaptiveFusionModel(freeze_clip=True).to(device)
    checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(Val AUROC: {checkpoint['val_auroc']:.4f})")

    plot_alpha_distribution(model, val_loader, device)
