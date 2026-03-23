import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloaders
from model import AdaptiveFusionModel


def collect_alpha(model, loader, device):
    model.eval()
    alphas, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            lbl = batch["label"].view(-1).cpu().numpy()
            _, alpha = model(input_ids, attention_mask, pixel_values)
            alpha_b = alpha.view(-1).cpu().numpy()
            alphas.extend(alpha_b.tolist())
            labels.extend(lbl.tolist())
    return np.array(alphas), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=str, default=None)
    parser.add_argument("--val-parquet", type=str, default=None)
    parser.add_argument("--static-checkpoint", type=str, default=None)
    parser.add_argument("--dynamic-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_p = args.train_parquet or str(root / "Data" / "train-00000-of-00001-6587b3a58d350036.parquet")
    val_p = args.val_parquet or str(root / "Data" / "validation-00000-of-00001-1508d9e5032c2c1f.parquet")
    static_ckpt = args.static_checkpoint or str(root / "checkpoints" / "best_model.pt")
    dynamic_ckpt = args.dynamic_checkpoint or str(root / "checkpoints" / "best_model_dynamic.pt")
    out = args.output or str(root / "results" / "alpha_static_dynamic_compare.png")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(train_p, val_p, batch_size=32)

    s_model = AdaptiveFusionModel(freeze_clip=True, use_dynamic=False).to(device)
    d_model = AdaptiveFusionModel(freeze_clip=True, use_dynamic=True).to(device)
    s = torch.load(static_ckpt, map_location=device)
    d = torch.load(dynamic_ckpt, map_location=device)
    s_model.load_state_dict(s["model_state"] if isinstance(s, dict) and "model_state" in s else s)
    d_model.load_state_dict(d["model_state"] if isinstance(d, dict) and "model_state" in d else d)

    s_alpha, labels = collect_alpha(s_model, val_loader, device)
    d_alpha, _ = collect_alpha(d_model, val_loader, device)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    bins = np.linspace(0, 1, 31)
    axes[0].hist(s_alpha, bins=bins, alpha=0.6, label="Static", color="steelblue")
    axes[0].hist(d_alpha, bins=bins, alpha=0.6, label="Dynamic", color="tomato")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Alpha Distribution: Static vs Dynamic")
    axes[0].legend()

    hate = labels == 1
    not_hate = labels == 0
    axes[1].boxplot(
        [s_alpha[not_hate], d_alpha[not_hate], s_alpha[hate], d_alpha[hate]],
        tick_labels=["S/Not", "D/Not", "S/Hate", "D/Hate"],
    )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Class-wise Alpha Comparison")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
