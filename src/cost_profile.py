"""
cost_profile.py
===============
Inference cost analysis for the dynamic and static V3 models, answering
Reviewers 2 & 3's missing computational-cost-analysis complaint:

  • Parameter breakdown per module — in particular the *delta* the fusion
    machinery (cross-modal attention + gating network + aux heads +
    classifier) adds on top of the vanilla CLIP backbone.
  • Per-sample inference latency in ms (batch size 1, median of repeated
    timed runs after warmup), plus amortised per-sample latency at the
    training batch size. For the dynamic model the latency is also split
    into CLIP feature extraction vs fusion+classification, which isolates
    the overhead of the proposed modules over a vanilla CLIP forward pass.
  • Estimated FLOPs per sample via torch.profiler (CPU pass, matmul/conv
    ops — a standard estimate, not an exact count), again split into
    full model vs CLIP-only so the fusion delta is explicit.

Weights don't affect cost, so models are profiled with random init unless
checkpoints are supplied; inputs are synthetic tensors with the correct
shapes, so this runs without the dataset present.

Usage
-----
    python cost_profile.py                     # profiles both models
    PROFILE_FLOPS=false python cost_profile.py # skip the (slow) CPU FLOPs pass
    RESULTS_DIR=... CLIP_MODEL=... also respected (see config.py)
"""

import os
import platform
import time

import numpy as np
import torch

from config import apply_env_overrides, base_config, get_device
from model  import AdaptiveFusionModelV3, StaticFusionModelV2


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def parameter_breakdown(model) -> dict:
    """Parameter counts grouped by top-level module, plus totals.
    'fusion_delta' = everything that is not the CLIP backbone, i.e. the
    parameter overhead of the proposed architecture over vanilla CLIP."""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()

    total     = sum(groups.values())
    clip      = groups.get("clip", 0)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "modules":      groups,
        "total":        total,
        "trainable":    trainable,
        "clip":         clip,
        "fusion_delta": total - clip,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LATENCY
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_batch(batch_size: int, device, seq_len: int = 77,
                     image_size: int = 224, vocab_size: int = 49408) -> dict:
    return {
        "input_ids":      torch.randint(0, vocab_size, (batch_size, seq_len),
                                        device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long,
                                     device=device),
        "pixel_values":   torch.randn(batch_size, 3, image_size, image_size,
                                      device=device),
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(fn, device, n_warmup: int = 3, n_runs: int = 30) -> dict:
    """Time fn() n_runs times after warmup. Returns ms stats (median/mean/std)."""
    for _ in range(n_warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms":   float(times.mean()),
        "std_ms":    float(times.std()),
        "n_runs":    n_runs,
    }


@torch.no_grad()
def profile_latency(model, device, is_dynamic: bool,
                    batch_sizes: tuple = (1, 8)) -> dict:
    """
    Latency at each batch size (ms/sample), plus — for the dynamic model —
    the split between CLIP feature extraction and fusion+classification.
    """
    model.eval()
    out = {}

    for bs in batch_sizes:
        batch = _synthetic_batch(bs, device)
        stats = measure_latency(
            lambda: model(batch["input_ids"], batch["attention_mask"],
                          batch["pixel_values"]),
            device,
        )
        out[f"bs{bs}"] = {**stats, "ms_per_sample": stats["median_ms"] / bs}

    if is_dynamic:
        batch = _synthetic_batch(1, device)
        feat_stats = measure_latency(
            lambda: model.extract_features(batch["input_ids"],
                                           batch["attention_mask"],
                                           batch["pixel_values"]),
            device,
        )
        txt, img = model.extract_features(batch["input_ids"],
                                          batch["attention_mask"],
                                          batch["pixel_values"])
        fuse_stats = measure_latency(
            lambda: model.fuse_and_classify(txt, img), device,
        )
        out["clip_features_bs1"] = feat_stats
        out["fusion_bs1"]        = fuse_stats

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs  (torch.profiler estimate — matmul/conv ops, CPU pass)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_flops(model, is_dynamic: bool) -> dict | None:
    """
    Per-sample FLOPs estimate via torch.profiler on a CPU forward pass
    (with_flops covers matmul/conv — the dominant terms in a transformer).
    Returns {'total': ..., 'clip_only': ..., 'fusion_delta': ...} or None.
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        model = model.to("cpu").eval()
        batch = _synthetic_batch(1, torch.device("cpu"))

        def _run_flops(fn):
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                fn()
            return sum(e.flops for e in prof.key_averages() if e.flops)

        total = _run_flops(
            lambda: model(batch["input_ids"], batch["attention_mask"],
                          batch["pixel_values"]))

        clip_only = None
        if is_dynamic:
            clip_only = _run_flops(
                lambda: model.extract_features(batch["input_ids"],
                                               batch["attention_mask"],
                                               batch["pixel_values"]))

        return {
            "total":        total,
            "clip_only":    clip_only,
            "fusion_delta": (total - clip_only) if clip_only else None,
        }
    except Exception as exc:                       # profiler support varies
        print(f"[FLOPs] estimate unavailable: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_params(n: int) -> str:
    return f"{n:,}  ({n/1e6:.2f} M)"


def _fmt_flops(n) -> str:
    return f"{n/1e9:.2f} GFLOPs" if n else "n/a"


def format_report(name: str, params: dict, latency: dict, flops) -> str:
    lines = [
        "─" * 68,
        f"{name}",
        "─" * 68,
        "Parameters:",
        f"  CLIP backbone      : {_fmt_params(params['clip'])}",
    ]
    for mod, n in sorted(params["modules"].items()):
        if mod != "clip":
            lines.append(f"  {mod:<19}: {_fmt_params(n)}")
    lines += [
        f"  TOTAL              : {_fmt_params(params['total'])}",
        f"  trainable          : {_fmt_params(params['trainable'])}",
        f"  Δ over vanilla CLIP: {_fmt_params(params['fusion_delta'])}"
        f"  = +{100 * params['fusion_delta'] / params['clip']:.2f}%",
        "Latency (median over repeated runs, synthetic inputs):",
    ]
    for key in ("bs1", "bs8"):
        if key in latency:
            s = latency[key]
            lines.append(
                f"  batch={key[2:]:<3} : {s['ms_per_sample']:8.2f} ms/sample   "
                f"({s['median_ms']:.2f} ms/batch ± {s['std_ms']:.2f})")
    if "clip_features_bs1" in latency:
        f_ms = latency["clip_features_bs1"]["median_ms"]
        g_ms = latency["fusion_bs1"]["median_ms"]
        lines += [
            f"  split @bs=1: CLIP features {f_ms:.2f} ms  |  "
            f"fusion+classify {g_ms:.2f} ms  "
            f"(fusion overhead ≈ {100 * g_ms / max(f_ms, 1e-9):.1f}% of CLIP cost)",
        ]
    if flops:
        lines.append("FLOPs per sample (profiler estimate, matmul/conv):")
        lines.append(f"  full model : {_fmt_flops(flops['total'])}")
        if flops.get("clip_only"):
            lines.append(f"  CLIP only  : {_fmt_flops(flops['clip_only'])}")
            lines.append(f"  Δ fusion   : {_fmt_flops(flops['fusion_delta'])}"
                         f"  = +{100 * flops['fusion_delta'] / flops['clip_only']:.2f}%")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = apply_env_overrides(base_config())
    results_dir  = os.environ.get("RESULTS_DIR", "../results")
    do_flops     = os.environ.get("PROFILE_FLOPS", "true").lower() in {
        "1", "true", "yes", "on"}

    device = get_device()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = f"{device.type} ({platform.machine()}, {platform.system()})"
    header = (f"Inference cost report\n"
              f"Device : {device_name}  |  torch {torch.__version__}\n"
              f"CLIP   : {cfg['clip_model']}")
    print(header + "\n")

    reports = [header]
    for name, cls, kwargs, is_dyn in (
        ("Dynamic V3 (AdaptiveFusionModelV3)", AdaptiveFusionModelV3, {}, True),
        ("Static V3 (StaticFusionModelV2, α=0.5)", StaticFusionModelV2,
         {"static_alpha": 0.5}, False),
    ):
        print(f"Profiling {name} …")
        model = cls(clip_model_name=cfg["clip_model"],
                    unfreeze_layers=cfg["unfreeze_layers"],
                    attn_dropout=cfg["attn_dropout"], **kwargs).to(device)

        params  = parameter_breakdown(model)
        latency = profile_latency(model, device, is_dynamic=is_dyn)
        flops   = estimate_flops(model, is_dynamic=is_dyn) if do_flops else None

        reports.append(format_report(name, params, latency, flops))
        del model

    report = "\n\n".join(reports)
    print("\n" + report)

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "cost_report.txt")
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
