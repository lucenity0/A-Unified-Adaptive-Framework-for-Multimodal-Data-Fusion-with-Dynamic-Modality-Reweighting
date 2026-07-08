"""
significance.py
===============
Statistical significance testing for the dynamic-vs-static comparison,
answering Reviewer 3's "statistical significance testing is not explicitly
reported."

Both tests operate on PAIRED per-sample validation predictions (both models
scored on the identical, un-shuffled validation set):

  1. Paired bootstrap (default 1,000 resamples, fixed seed):
       • per-model 95% percentile CIs for AUROC / accuracy / macro-F1
         (for Table 3 "± CI" columns);
       • CI + two-sided p-value for the *deltas* (Δ = dynamic − static).
         Pairing = each resample draws one set of indices applied to both
         models, so sample-difficulty variance cancels out of the delta.

  2. McNemar's test on thresholded predictions (each model at its own
     validation-selected operating threshold):
       exact binomial test when discordant pairs < 25, otherwise the
       continuity-corrected chi-square version. AUROC has no per-sample
       right/wrong, so McNemar covers the accuracy claim and the bootstrap
       covers the AUROC claim.

Usage
-----
    python significance.py            # loads both best checkpoints, evaluates
                                      # on the validation split, prints + saves
                                      # the report

    # or programmatically on already-collected predictions:
    from significance import compare_models, format_report
    results = compare_models(labels, probs_dynamic, probs_static,
                             thresh_a=0.55, thresh_b=0.50)

Env overrides: DYNAMIC_CKPT / STATIC_CKPT (full paths), plus the usual
DATA_DIR / VAL_PARQUET / CHECKPOINT_DIR from config.py.
"""

import os

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# CORE TESTS  (pure numpy/scipy — no torch needed)
# ─────────────────────────────────────────────────────────────────────────────


def _metrics(labels, probs, thresh):
    preds = (probs >= thresh).astype(int)
    return {
        "auroc":    roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="macro"),
    }


def paired_bootstrap(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    thresh_a: float = 0.5,
    thresh_b: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    Paired bootstrap over validation samples. Model A is the proposed model
    (dynamic), model B the baseline (static); deltas are A − B.

    Returns a dict with, per metric:
        a_mean/a_lo/a_hi, b_mean/b_lo/b_hi   — per-model point est. + CI
        delta/delta_lo/delta_hi/p_value      — paired delta CI + two-sided p
    """
    rng = np.random.default_rng(seed)
    n   = len(labels)

    boots = {m: {"a": [], "b": []} for m in ("auroc", "accuracy", "f1")}

    done = 0
    while done < n_boot:
        idx = rng.integers(0, n, size=n)
        y   = labels[idx]
        if len(np.unique(y)) < 2:      # degenerate resample — AUROC undefined
            continue
        ma = _metrics(y, probs_a[idx], thresh_a)
        mb = _metrics(y, probs_b[idx], thresh_b)
        for m in boots:
            boots[m]["a"].append(ma[m])
            boots[m]["b"].append(mb[m])
        done += 1

    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    point_a = _metrics(labels, probs_a, thresh_a)
    point_b = _metrics(labels, probs_b, thresh_b)

    out = {"n_boot": n_boot, "seed": seed, "ci": ci}
    for m in boots:
        a = np.array(boots[m]["a"])
        b = np.array(boots[m]["b"])
        d = a - b
        # two-sided bootstrap p-value with add-one continuity correction
        p_le = (np.sum(d <= 0) + 1) / (n_boot + 1)
        p_ge = (np.sum(d >= 0) + 1) / (n_boot + 1)
        out[m] = {
            "a":        point_a[m],
            "a_lo":     float(np.percentile(a, lo_q)),
            "a_hi":     float(np.percentile(a, hi_q)),
            "b":        point_b[m],
            "b_lo":     float(np.percentile(b, lo_q)),
            "b_hi":     float(np.percentile(b, hi_q)),
            "delta":    point_a[m] - point_b[m],
            "delta_lo": float(np.percentile(d, lo_q)),
            "delta_hi": float(np.percentile(d, hi_q)),
            "p_value":  float(min(1.0, 2 * min(p_le, p_ge))),
        }
    return out


def mcnemar_test(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """
    McNemar's test on paired binary predictions.

    n01 = samples model A gets right and model B gets wrong
    n10 = samples model A gets wrong and model B gets right

    Exact binomial test when n01 + n10 < 25; otherwise chi-square with
    Edwards continuity correction.
    """
    correct_a = preds_a == labels
    correct_b = preds_b == labels

    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {"n01": n01, "n10": n10, "method": "degenerate", "p_value": 1.0}

    if n_discordant < 25:
        method = "exact binomial"
        p = stats.binomtest(min(n01, n10), n_discordant, 0.5,
                            alternative="two-sided").pvalue
        statistic = None
    else:
        method = "chi-square (continuity-corrected)"
        statistic = (abs(n01 - n10) - 1) ** 2 / n_discordant
        p = float(stats.chi2.sf(statistic, df=1))

    return {
        "n01": n01, "n10": n10,
        "method": method,
        "statistic": statistic,
        "p_value": float(p),
    }


def compare_models(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    thresh_a: float = 0.5,
    thresh_b: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Run both tests. Model A = dynamic (proposed), model B = static (baseline)."""
    boot = paired_bootstrap(
        labels, probs_a, probs_b, thresh_a, thresh_b, n_boot=n_boot, seed=seed
    )
    mcn = mcnemar_test(
        labels,
        (probs_a >= thresh_a).astype(int),
        (probs_b >= thresh_b).astype(int),
    )
    return {"bootstrap": boot, "mcnemar": mcn,
            "thresh_a": thresh_a, "thresh_b": thresh_b, "n": len(labels)}


def format_report(results: dict,
                  name_a: str = "Dynamic V3",
                  name_b: str = "Static V3") -> str:
    b   = results["bootstrap"]
    mcn = results["mcnemar"]
    ci_pct = int(b["ci"] * 100)

    lines = [
        "=" * 72,
        f"Significance report — {name_a} vs {name_b} "
        f"(n={results['n']}, {b['n_boot']} bootstrap resamples, seed {b['seed']})",
        f"Operating thresholds: {name_a} @ {results['thresh_a']:.2f}, "
        f"{name_b} @ {results['thresh_b']:.2f}",
        "=" * 72,
    ]
    for m, label in (("auroc", "AUROC"), ("accuracy", "Accuracy"), ("f1", "Macro F1")):
        r = b[m]
        lines += [
            f"{label}:",
            f"  {name_a:12s}: {r['a']:.4f}  ({ci_pct}% CI [{r['a_lo']:.4f}, {r['a_hi']:.4f}])",
            f"  {name_b:12s}: {r['b']:.4f}  ({ci_pct}% CI [{r['b_lo']:.4f}, {r['b_hi']:.4f}])",
            f"  Δ (paired)  : {r['delta']:+.4f}  ({ci_pct}% CI [{r['delta_lo']:+.4f}, "
            f"{r['delta_hi']:+.4f}])   p = {r['p_value']:.4f}"
            f"{'  *' if r['p_value'] < 0.05 else ''}",
        ]
    stat = (f"{mcn['statistic']:.3f}" if mcn.get("statistic") is not None else "—")
    lines += [
        f"McNemar ({mcn['method']}):",
        f"  discordant pairs: {name_a} right/{name_b} wrong = {mcn['n01']}, "
        f"{name_a} wrong/{name_b} right = {mcn['n10']}",
        f"  statistic = {stat}   p = {mcn['p_value']:.4f}"
        f"{'  *' if mcn['p_value'] < 0.05 else ''}",
        "=" * 72,
        "* significant at α = 0.05 (two-sided)",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUNNER  (loads both best checkpoints, evaluates, reports)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor

    from config      import apply_env_overrides, base_config, get_device
    from dataset     import HatefulMemesDatasetV2
    from model       import AdaptiveFusionModelV3, StaticFusionModelV2
    from train_utils import collect_predictions

    cfg = apply_env_overrides(base_config())
    dynamic_ckpt = os.environ.get(
        "DYNAMIC_CKPT", os.path.join(cfg["checkpoint_dir"], "best_model_dynamic_v3.pt"))
    static_ckpt  = os.environ.get(
        "STATIC_CKPT",  os.path.join(cfg["checkpoint_dir"], "best_model_static_v3.pt"))
    results_dir  = os.environ.get("RESULTS_DIR", "../results")

    device = get_device()
    print(f"Device : {device}")

    processor  = CLIPProcessor.from_pretrained(cfg["clip_model"])
    val_ds     = HatefulMemesDatasetV2(cfg["val_parquet"], processor, augment=False,
                                       max_samples=cfg.get("max_samples", 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers", 2),
                            pin_memory=(device.type == "cuda"))

    # ── dynamic model ─────────────────────────────────────────────────────────
    dyn_ck  = torch.load(dynamic_ckpt, map_location=device, weights_only=False)
    dyn_cfg = dyn_ck.get("config", {})
    dyn = AdaptiveFusionModelV3(
        clip_model_name=dyn_cfg.get("clip_model", cfg["clip_model"]),
        unfreeze_layers=dyn_cfg.get("unfreeze_layers", 4),
        attn_dropout=dyn_cfg.get("attn_dropout", 0.2),
    ).to(device)
    dyn.load_state_dict(dyn_ck["model_state"])
    thresh_dyn = float(dyn_ck.get("opt_threshold", 0.5))
    print(f"Loaded dynamic ckpt (epoch {dyn_ck.get('epoch', '?')}, "
          f"val AUROC {dyn_ck.get('val_auroc', float('nan')):.4f}, "
          f"threshold {thresh_dyn:.2f})")

    # ── static model ──────────────────────────────────────────────────────────
    sta_ck  = torch.load(static_ckpt, map_location=device, weights_only=False)
    sta_cfg = sta_ck.get("config", {})
    sta = StaticFusionModelV2(
        clip_model_name=sta_cfg.get("clip_model", cfg["clip_model"]),
        unfreeze_layers=sta_cfg.get("unfreeze_layers", 4),
        static_alpha=sta_cfg.get("static_alpha", 0.5),
        attn_dropout=sta_cfg.get("attn_dropout", 0.2),
    ).to(device)
    sta.load_state_dict(sta_ck["model_state"])
    thresh_sta = float(sta_ck.get("opt_threshold", 0.5))
    print(f"Loaded static ckpt  (epoch {sta_ck.get('epoch', '?')}, "
          f"val AUROC {sta_ck.get('val_auroc', float('nan')):.4f}, "
          f"threshold {thresh_sta:.2f})")

    # ── paired predictions on the identical validation order ─────────────────
    print("\nCollecting paired validation predictions …")
    probs_dyn, labels  = collect_predictions(dyn, val_loader, device, is_dynamic=True)
    probs_sta, labels2 = collect_predictions(sta, val_loader, device, is_dynamic=False)
    assert np.array_equal(labels, labels2), "validation order mismatch — pairing broken"

    results = compare_models(labels, probs_dyn, probs_sta,
                             thresh_a=thresh_dyn, thresh_b=thresh_sta)
    report = format_report(results)
    print("\n" + report)

    # ── persist report + paired predictions for reproducibility ──────────────
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "significance_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    preds_path = os.path.join(results_dir, "paired_val_predictions.csv")
    pd.DataFrame({
        "label":        labels,
        "prob_dynamic": probs_dyn,
        "prob_static":  probs_sta,
    }).to_csv(preds_path, index=False)
    print(f"\nSaved: {report_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()
