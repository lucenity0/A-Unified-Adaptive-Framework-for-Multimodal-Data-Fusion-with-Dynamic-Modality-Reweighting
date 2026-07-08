"""
dataset.py
==========
HatefulMemesDatasetV2 — standalone module.

V3 adds an optional `augment` flag for the training split:
  - Random horizontal flip of image (p=0.3)
  - Random colour jitter (brightness/contrast, p=0.3)
  - Random text-token dropout (replaces up to 10 % of non-special tokens with
    the [UNK] id, probability per token = 0.1)

These augmentations are applied *before* the CLIP processor so that the
processor's normalization/resizing still runs last.  Pass augment=False
(default) for the validation / test split.
"""

import io
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance


class HatefulMemesDatasetV2(torch.utils.data.Dataset):
    """
    Dataset for the Hateful Memes Challenge stored as a Parquet file.

    Expected schema
    ---------------
    image  : bytes / dict{"bytes": bytes} / path str
    text   : str
    label  : int  (0 = benign, 1 = hateful)   — optional for test sets
    """

    def __init__(
        self,
        parquet_path: str,
        processor,
        max_text_length: int = 77,
        augment: bool = False,
        max_samples: int = 0,
    ):
        self.df = pd.read_parquet(parquet_path)
        if max_samples and max_samples < len(self.df):
            # random (seeded) subsample — used for smoke tests, 0 = full set
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        self.processor  = processor
        self.max_len    = max_text_length
        self.augment    = augment
        print(f"[Dataset] Loaded {len(self.df)} samples from {parquet_path}"
              f"  |  augment={augment}")

    # ── length ────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    # ── single item ───────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── load image ────────────────────────────────────────────────────────
        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            image = Image.open(str(img_data)).convert("RGB")

        # ── load text ─────────────────────────────────────────────────────────
        text  = str(row["text"]) if pd.notna(row["text"]) else ""
        label = (
            int(row["label"])
            if "label" in row.index and pd.notna(row["label"])
            else -1
        )

        # ── optional augmentation (training only) ─────────────────────────────
        if self.augment:
            image = self._augment_image(image)
            text  = self._augment_text(text)

        # ── CLIP processor ────────────────────────────────────────────────────
        enc = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float32),
        }

    # ── augmentation helpers ──────────────────────────────────────────────────
    @staticmethod
    def _augment_image(image: Image.Image) -> Image.Image:
        """Light colour / geometry augmentation — keeps semantic content intact."""
        if random.random() < 0.3:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            image  = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            image  = ImageEnhance.Contrast(image).enhance(factor)
        return image

    @staticmethod
    def _augment_text(text: str) -> str:
        """Randomly drop ~10 % of words to improve robustness."""
        if not text:
            return text
        words = text.split()
        words = [w for w in words if random.random() > 0.10]
        return " ".join(words) if words else text
