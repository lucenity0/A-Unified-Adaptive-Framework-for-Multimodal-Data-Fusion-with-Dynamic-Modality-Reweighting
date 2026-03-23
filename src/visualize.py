"""
visualize.py
============
Visualize modality weights (alpha) on validation set.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloaders
from model import AdaptiveFusionModel


def plot_alpha_distribution(model, val_loader, device, save_path="alpha_distribution.png"):
    model.eval()
    all_alphas, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]

            _, alpha = model(input_ids, attention_mask, pixel_values)
            alpha = alpha.detach().float()
            alpha_batch = alpha.view(-1).cpu().numpy()

            all_alphas.extend(alpha_batch.tolist())
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())

    all_alphas = np.array(all_alphas)
    all_labels = np.array(all_labels)

    hateful_alphas = all_alphas[all_labels == 1]
    not_hateful_alphas = all_alphas[all_labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(not_hateful_alphas, bins=30, alpha=0.7,
                 label=f"Not Hateful (n={len(not_hateful_alphas)})", color="steelblue")
    axes[0].hist(hateful_alphas, bins=30, alpha=0.7,
                 label=f"Hateful (n={len(hateful_alphas)})", color="tomato")
    axes[0].set_xlabel("Mean Alpha\n← Text dominant        Image dominant →", fontsize=11)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dynamic Modality Weight Distribution")
    axes[0].legend()
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)

    axes[1].boxplot(
        [not_hateful_alphas, hateful_alphas],
        tick_labels=["Not Hateful", "Hateful"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.7),
    )
    axes[1].set_ylabel("Mean Alpha (Image Weight)")
    axes[1].set_title("Alpha Distribution: Hateful vs Not Hateful")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {save_path}")
    if len(hateful_alphas) > 0:
        print(f"Hateful mean±std: {hateful_alphas.mean():.3f} ± {hateful_alphas.std():.3f}")
    if len(not_hateful_alphas) > 0:
        print(f"Not hateful mean±std: {not_hateful_alphas.mean():.3f} ± {not_hateful_alphas.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=str, default=None)
    parser.add_argument("--val-parquet", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    train_parquet = Path(args.train_parquet) if args.train_parquet else root_dir / "Data" / "train-00000-of-00001-6587b3a58d350036.parquet"
    val_parquet = Path(args.val_parquet) if args.val_parquet else root_dir / "Data" / "validation-00000-of-00001-1508d9e5032c2c1f.parquet"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else root_dir / "checkpoints" / "best_model_dynamic.pt"
    output_path = Path(args.output) if args.output else root_dir / "results" / "alpha_distribution_dynamic.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    _, val_loader = get_dataloaders(str(train_parquet), str(val_parquet), batch_size=args.batch_size)
    model = AdaptiveFusionModel(freeze_clip=True, use_dynamic=True).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    plot_alpha_distribution(model, val_loader, device, save_path=str(output_path))
