"""
demo.py
=======
Interactive demo for the Hateful Memes classifier.
Type any meme caption (+ optional image path) and get a prediction.

Run:
    python3 demo.py

Controls:
    - Enter text when prompted
    - Optionally provide an image file path
    - Press ENTER to skip image (uses blank white image)
    - Type 'quit' or 'exit' to stop
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor
from PIL import Image
from model import AdaptiveFusionModel
import io
import os

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "../checkpoints/best_model.pt"


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL (once at startup)
# ─────────────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    model = AdaptiveFusionModel(freeze_clip=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint


# ─────────────────────────────────────────────────────────────────
# PREDICT SINGLE SAMPLE
# ─────────────────────────────────────────────────────────────────
def predict(text, image, model, processor, device):
    """
    Run inference on one text + image pair.
    Returns prob_hateful, predicted_label, alpha, confidence
    """
    encoding = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    pixel_values   = encoding['pixel_values'].to(device)

    with torch.no_grad():
        logit, alpha = model(input_ids, attention_mask, pixel_values)

    prob  = torch.sigmoid(logit).item()
    label = 1 if prob >= 0.5 else 0
    conf  = prob if prob >= 0.5 else 1 - prob
    alpha_val = alpha.item() if hasattr(alpha, 'item') else float(alpha)

    return prob, label, alpha_val, conf


# ─────────────────────────────────────────────────────────────────
# PRINT RESULT
# ─────────────────────────────────────────────────────────────────
def print_result(text, image_path, prob, label, alpha_val, conf):
    print()
    print("=" * 55)
    print("  PREDICTION RESULT")
    print("=" * 55)
    print(f"  Text         : {text}")
    print(f"  Image        : {image_path if image_path else 'None (blank used)'}")
    print("-" * 55)

    verdict = "HATEFUL" if label == 1 else "NOT HATEFUL"
    emoji   = "X" if label == 1 else "OK"

    print(f"  Verdict      : [{emoji}] {verdict}")
    print(f"  Probability  : {prob:.4f}  ({prob*100:.1f}% hateful)")
    print(f"  Confidence   : {conf*100:.1f}%")
    print(f"  Alpha        : {alpha_val:.4f}  ({'image' if alpha_val > 0.5 else 'text'} dominant)")

    # Visual probability bar
    bar_len   = 40
    filled    = int(prob * bar_len)
    bar       = "#" * filled + "-" * (bar_len - filled)
    print(f"  Hate score   : [{bar}] {prob:.2f}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 55)
    print("  Hateful Meme Classifier - Interactive Demo")
    print("  Phase 1: Cross-Attn + Static Reweighting")
    print("=" * 55)
    print(f"  Device : {device}")
    print("  Loading model...")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n  ERROR: No checkpoint found at {CHECKPOINT_PATH}")
        print("  Run train.py first to generate a checkpoint.")
        return

    model, checkpoint = load_model(CHECKPOINT_PATH, device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(Val AUROC: {checkpoint['val_auroc']:.4f})")
    print()
    print("  Instructions:")
    print("  - Enter meme text when prompted")
    print("  - Optionally enter an image file path")
    print("  - Press ENTER to skip image (white blank used)")
    print("  - Type 'quit' or 'exit' to stop")
    print("=" * 55)

    # Blank fallback image (white 224x224)
    blank_image = Image.new("RGB", (224, 224), color=(255, 255, 255))

    while True:
        print()

        # ── Get text input ──────────────────────────────────────
        try:
            text = input("  Enter meme text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting demo.")
            break

        if text.lower() in ('quit', 'exit', 'q'):
            print("  Exiting demo.")
            break

        if not text:
            print("  Text cannot be empty. Try again.")
            continue

        # ── Get optional image ──────────────────────────────────
        try:
            image_path = input("  Image path (or ENTER to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting demo.")
            break

        if image_path:
            if not os.path.exists(image_path):
                print(f"  Image not found: {image_path} -- using blank image instead.")
                image      = blank_image
                image_path = None
            else:
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"  Could not open image ({e}) -- using blank image instead.")
                    image      = blank_image
                    image_path = None
        else:
            image      = blank_image
            image_path = None

        # ── Run prediction ──────────────────────────────────────
        prob, label, alpha_val, conf = predict(
            text, image, model, processor, device
        )

        print_result(text, image_path, prob, label, alpha_val, conf)


if __name__ == "__main__":
    main()
