import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloaders
from model import AdaptiveFusionModel


def plot_alpha_distribution(model, val_loader, device, save_path):
    model.eval()
    all_alphas, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]
            _, alpha = model(input_ids, attention_mask, pixel_values)
            all_alphas.extend(alpha.view(-1).cpu().numpy().tolist())
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())

    all_alphas = np.array(all_alphas)
    all_labels = np.array(all_labels)
    hateful_alphas = all_alphas[all_labels == 1]
    not_hateful_alphas = all_alphas[all_labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(not_hateful_alphas, bins=30, alpha=0.7, label="Not Hateful", color="steelblue")
    axes[0].hist(hateful_alphas, bins=30, alpha=0.7, label="Hateful", color="tomato")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Static Alpha Distribution")
    axes[0].legend()

    axes[1].boxplot([not_hateful_alphas, hateful_alphas], tick_labels=["Not Hateful", "Hateful"])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Static Alpha Boxplot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=str, default=None)
    parser.add_argument("--val-parquet", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_p = args.train_parquet or str(root / "Data" / "train-00000-of-00001-6587b3a58d350036.parquet")
    val_p = args.val_parquet or str(root / "Data" / "validation-00000-of-00001-1508d9e5032c2c1f.parquet")
    ckpt = args.checkpoint or str(root / "checkpoints" / "best_model.pt")
    out = args.output or str(root / "results" / "alpha_distribution_static.png")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(train_p, val_p, batch_size=32)
    model = AdaptiveFusionModel(freeze_clip=True, use_dynamic=False).to(device)
    c = torch.load(ckpt, map_location=device)
    model.load_state_dict(c["model_state"] if isinstance(c, dict) and "model_state" in c else c)
    plot_alpha_distribution(model, val_loader, device, out)
