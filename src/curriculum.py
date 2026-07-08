"""
curriculum.py
=============
Difficulty-ordered curriculum for Hateful Memes training (from DynCIM).

The Hateful Memes dataset is adversarially curated: benign images with hateful
text, or hateful imagery with innocent text.  Exposing the model to hard
confounders from epoch 1 destabilises early training.

Strategy
--------
Use CLIP's zero-shot cosine similarity between the image embedding and the
text embedding as a proxy for *difficulty*.

  • Low cosine similarity  →  the two modalities tell very different stories
    (image and text are incongruent) — these tend to be HARDER hateful memes.
  • High cosine similarity →  the modalities align (easy benign memes where
    both image and text are innocuous).

Phase schedule (default)
------------------------
  Phase 1  epochs 1 – phase_1_end     : easy samples only (sim ≥ easy_thresh)
  Phase 2  epochs phase_1_end + 1 …   : all samples

Usage in train_dynamic_v3.py
-----------------------------
    from curriculum import build_curriculum_loaders

    phase1_loader, full_loader = build_curriculum_loaders(
        train_dataset, processor, device, cfg
    )

    for epoch in range(1, cfg["num_epochs"] + 1):
        loader = phase1_loader if epoch <= cfg["curriculum_phase1_end"] else full_loader
        train_one_epoch(model, loader, ...)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import CLIPModel

from clip_features import extract_clip_image_features, extract_clip_text_features


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_clip_similarity(dataset, clip_model_name: str, device,
                            batch_size: int = 32, num_workers: int = 2) -> np.ndarray:
    """
    Compute per-sample CLIP cosine similarity between image and text embeddings
    for every sample in `dataset`.

    Parameters
    ----------
    dataset        : HatefulMemesDatasetV2 (augment should be False)
    clip_model_name: e.g. "openai/clip-vit-large-patch14"
    device         : torch.device
    batch_size     : number of samples per CLIP inference batch

    Returns
    -------
    similarities : np.ndarray  shape (N,)  values in [-1, 1]
    """
    print("[Curriculum] Computing CLIP cosine similarities …")
    clip = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    sims   = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)

        txt = extract_clip_text_features(
            clip, input_ids=input_ids, attention_mask=attention_mask
        )
        img = extract_clip_image_features(clip, pixel_values=pixel_values)

        txt = torch.nn.functional.normalize(txt, dim=-1)
        img = torch.nn.functional.normalize(img, dim=-1)

        cos = (txt * img).sum(dim=-1)   # (B,)
        sims.extend(cos.cpu().numpy())

    del clip
    torch.cuda.empty_cache() if device.type == "cuda" else None

    similarities = np.array(sims)
    print(f"[Curriculum] Similarity stats: mean={similarities.mean():.3f}  "
          f"std={similarities.std():.3f}  "
          f"min={similarities.min():.3f}  max={similarities.max():.3f}")
    return similarities


# ─────────────────────────────────────────────────────────────────────────────
# LOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_curriculum_loaders(
    train_dataset,
    clip_model_name: str,
    device,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (phase1_loader, full_loader).

    phase1_loader  : easy subset — samples with CLIP cosine sim ≥ easy_thresh
    full_loader    : all training samples (standard shuffled DataLoader)

    Config keys used
    ----------------
    cfg["clip_model"]               : CLIP model name
    cfg["batch_size"]               : DataLoader batch size
    cfg["curriculum_easy_thresh"]   : cosine-sim floor for easy subset (default 0.25)
    cfg["curriculum_phase1_end"]    : last epoch of easy-only phase (default 3)
    """
    easy_thresh = cfg.get("curriculum_easy_thresh", 0.25)

    # ── compute similarities ──────────────────────────────────────────────────
    sims = compute_clip_similarity(
        train_dataset, clip_model_name, device,
        batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("num_workers", 2),
    )

    # ── easy subset ───────────────────────────────────────────────────────────
    easy_indices = np.where(sims >= easy_thresh)[0].tolist()
    easy_subset  = Subset(train_dataset, easy_indices)

    print(f"[Curriculum] Phase-1 easy subset: {len(easy_indices)} / {len(train_dataset)} samples "
          f"(sim ≥ {easy_thresh})")

    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )

    phase1_loader = DataLoader(easy_subset,       shuffle=True, **loader_kwargs)
    full_loader   = DataLoader(train_dataset,     shuffle=True, **loader_kwargs)

    return phase1_loader, full_loader
