"""
dataset.py
==========
Loads the Hateful Memes dataset directly from Parquet files.
No conversion needed — reads image bytes and text on the fly.
"""

import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


class HatefulMemesDataset(Dataset):
    """
    Reads directly from .parquet files downloaded from HuggingFace.

    Expected columns: 'image', 'text', 'label'
    - image : dict with 'bytes' key (PNG bytes)
    - text  : string
    - label : 0 or 1
    """
    def __init__(self, parquet_path, processor, max_text_length=77):
        self.df              = pd.read_parquet(parquet_path)
        self.processor       = processor
        self.max_text_length = max_text_length

        print(f"Loaded {len(self.df)} samples from {parquet_path}")
        print(f"Columns: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load Image ────────────────────────────────────────────
        img_data = row['image']

        if isinstance(img_data, dict) and 'bytes' in img_data:
            image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            image = Image.open(str(img_data)).convert("RGB")

        # ── Load Text ─────────────────────────────────────────────
        text = str(row['text']) if pd.notna(row['text']) else ""

        # ── Load Label ────────────────────────────────────────────
        label = int(row['label']) if 'label' in row.index and pd.notna(row['label']) else -1

        # ── Preprocess with CLIP ──────────────────────────────────
        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values':   encoding['pixel_values'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.float32)
        }


def get_dataloaders(train_parquet, val_parquet, batch_size=16, num_workers=2):
    """
    Returns train and validation DataLoaders.

    Args:
        train_parquet : path to train parquet file
        val_parquet   : path to validation parquet file
        batch_size    : 16 recommended for M2 Mac
        num_workers   : 2 is safe on Mac
    """
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = HatefulMemesDataset(train_parquet, processor)
    val_dataset   = HatefulMemesDataset(val_parquet,   processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False    # MPS does not support pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False    # MPS does not support pin_memory
    )

    return train_loader, val_loader
