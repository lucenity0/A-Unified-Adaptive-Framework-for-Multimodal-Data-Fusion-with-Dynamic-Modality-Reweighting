import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloaders
from model import AdaptiveFusionModel


def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def collect(model, loader, device):
    model.eval()
    alphas, probs, labels = [], [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            pv = b["pixel_values"].to(device)
            l = b["label"].view(-1).cpu().numpy()
            logits, alpha = model(ids, am, pv)
            p = torch.sigmoid(logits).view(-1).cpu().numpy()
            a = alpha.view(-1).cpu().numpy()
            alphas.extend(a.tolist())
            probs.extend(p.tolist())
            labels.extend(l.tolist())
    return np.array(alphas), np.array(probs), np.array(labels).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_p = str(root / "Data" / "train-00000-of-00001-6587b3a58d350036.parquet")
    val_p = str(root / "Data" / "validation-00000-of-00001-1508d9e5032c2c1f.parquet")
    out = args.output or str(root / "results" / "alpha_advanced_static_dynamic.png")
    s_ckpt = str(root / "checkpoints" / "best_model.pt")
    d_ckpt = str(root / "checkpoints" / "best_model_dynamic.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    _, loader = get_dataloaders(train_p, val_p, batch_size=32)
    sm = AdaptiveFusionModel(freeze_clip=True, use_dynamic=False).to(device)
    dm = AdaptiveFusionModel(freeze_clip=True, use_dynamic=True).to(device)
    s = torch.load(s_ckpt, map_location=device)
    d = torch.load(d_ckpt, map_location=device)
    sm.load_state_dict(s["model_state"] if isinstance(s, dict) and "model_state" in s else s)
    dm.load_state_dict(d["model_state"] if isinstance(d, dict) and "model_state" in d else d)

    sa, sp, y = collect(sm, loader, device)
    da, dp, y2 = collect(dm, loader, device)
    if not np.array_equal(y, y2):
        raise RuntimeError("Label mismatch.")

    fig, ax = plt.subplots(1, 3, figsize=(21, 5))
    mask0, mask1 = y == 0, y == 1
    for vals, name, c, ls in [
        (sa[mask0], "Static/Not", "steelblue", "-"),
        (da[mask0], "Dynamic/Not", "steelblue", "--"),
        (sa[mask1], "Static/Hate", "tomato", "-"),
        (da[mask1], "Dynamic/Hate", "tomato", "--"),
    ]:
        if len(vals):
            x, e = ecdf(vals)
            ax[0].plot(x, e, label=name, color=c, linestyle=ls)
    ax[0].set_title("ECDF alpha")
    ax[0].legend(fontsize=8)
    ax[0].axvline(0.5, color="black", linestyle="--", linewidth=1)

    groups = [sa[mask0], da[mask0], sa[mask1], da[mask1]]
    vp = ax[1].violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], ["steelblue", "steelblue", "tomato", "tomato"]):
        body.set_facecolor(c); body.set_alpha(0.35)
    ax[1].boxplot(groups, tick_labels=["S/Not", "D/Not", "S/Hate", "D/Hate"], widths=0.2)
    ax[1].set_title("Violin+Box alpha")
    ax[1].axhline(0.5, color="black", linestyle="--", linewidth=1)

    bins = np.linspace(0, 1, 13)
    for a, p, c, lbl in [(sa, sp, "steelblue", "Static"), (da, dp, "tomato", "Dynamic")]:
        centers = 0.5 * (bins[:-1] + bins[1:])
        means = []
        for i in range(len(bins)-1):
            m = (a >= bins[i]) & (a < bins[i+1] if i < len(bins)-2 else a <= bins[i+1])
            means.append(float(p[m].mean()) if m.any() else np.nan)
        means = np.array(means)
        ok = ~np.isnan(means)
        ax[2].plot(centers[ok], means[ok], marker="o", color=c, label=lbl)
    ax[2].set_title("Alpha vs hateful prob")
    ax[2].set_ylim(0, 1)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
