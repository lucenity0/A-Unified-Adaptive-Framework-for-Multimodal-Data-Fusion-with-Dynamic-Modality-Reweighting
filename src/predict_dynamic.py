"""
predict_dynamic.py
==================
Runs inference on the test set with Dynamic Reweighting (Phase 2).
Saves predictions to results/test_predictions_dynamic.csv
"""

import os
import torch
import numpy as np
import pandas as pd
from dataset import HatefulMemesDataset
from model import AdaptiveFusionModel
from transformers import CLIPProcessor
from torch.utils.data import DataLoader


TEST_PARQUET = "../Data/test/train-00000-of-00001-19a6f88cedb64664.parquet"
CHECKPOINT_PATH = "../checkpoints/best_model_dynamic.pt"
RESULTS_DIR = "../results"
BATCH_SIZE = 16


class TestDataset(HatefulMemesDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        import io
        from PIL import Image

        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            image = Image.open(str(img_data)).convert("RGB")

        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = -1

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        sample_id = row["id"] if "id" in row.index else idx
        if torch.is_tensor(sample_id):
            sample_id = sample_id.item()
        elif hasattr(sample_id, "item"):
            sample_id = sample_id.item()
        sample_id = str(sample_id)
        if sample_id.startswith("tensor(") and sample_id.endswith(")"):
            sample_id = sample_id[7:-1]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "id": sample_id,
            "text": text
        }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = AdaptiveFusionModel(freeze_clip=True, use_dynamic=True).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    if isinstance(checkpoint, dict) and "epoch" in checkpoint and "val_auroc" in checkpoint:
        print(f"Loaded dynamic model from epoch {checkpoint['epoch']} "
              f"(Val AUROC: {checkpoint['val_auroc']:.4f})")
    else:
        print("Loaded dynamic model checkpoint.")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    test_dataset = TestDataset(TEST_PARQUET, processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    all_ids, all_probs, all_preds, all_alphas, all_texts = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            logits, alpha = model(input_ids, attention_mask, pixel_values)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            alpha_means = alpha.cpu().numpy().reshape(-1)

            all_ids.extend(batch["id"])
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_alphas.extend(alpha_means)
            all_texts.extend(batch["text"])

    results_df = pd.DataFrame({
        "id": all_ids,
        "text": all_texts,
        "prob_hateful": np.round(all_probs, 4),
        "predicted_label": all_preds,
        "confidence": [round(p if p >= 0.5 else 1 - p, 4) for p in all_probs],
        "dominant_modality": ["image" if a > 0.5 else "text" for a in all_alphas],
        "alpha_mean": np.round(all_alphas, 4)
    })

    save_path = os.path.join(RESULTS_DIR, "test_predictions_dynamic.csv")
    results_df.to_csv(save_path, index=False)

    total = len(results_df)
    hateful = results_df["predicted_label"].sum()
    img_dom = (results_df["dominant_modality"] == "image").sum()
    txt_dom = (results_df["dominant_modality"] == "text").sum()
    print(f"\nTest Inference Complete -- {total} samples")
    print(f"  Predicted Hateful     : {hateful} ({100*hateful/total:.1f}%)")
    print(f"  Predicted Not Hateful : {total-hateful} ({100*(total-hateful)/total:.1f}%)")
    print(f"  Image-dominant memes  : {img_dom} ({100*img_dom/total:.1f}%)")
    print(f"  Text-dominant  memes  : {txt_dom} ({100*txt_dom/total:.1f}%)")
    print(f"\nSaved --> {save_path}")


if __name__ == "__main__":
    main()
