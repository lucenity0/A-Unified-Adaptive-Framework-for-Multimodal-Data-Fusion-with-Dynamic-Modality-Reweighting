#!/bin/bash
# Build hateful_memes_colab_bundle.zip for the Colab/Kaggle notebook.
# Contents: src/*.py + Data/*.parquet (code + dataset, no checkpoints).
# Run from the repo root:  bash colab/make_bundle.sh
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="colab/hateful_memes_colab_bundle.zip"

rm -f "$OUT"
zip -q "$OUT" src/*.py Data/*.parquet
echo "Built $OUT:"
unzip -l "$OUT"
