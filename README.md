# A Unified Adaptive Framework for Multimodal Data Fusion with Dynamic Modality Reweighting

Multimodal hateful meme detection built on a **CLIP ViT-L/14** backbone, a
two-layer bidirectional **cross-modal attention** module, and a **dynamic
gating network** that learns a per-sample scalar weight α ∈ (0, 1) balancing
the visual and textual contribution to the final classification — instead of
the fixed equal-weight fusion used by most prior work.

The repository contains two comparable models:

- **`AdaptiveFusionModelV3`** — the proposed dynamic model. Adds
  uncertainty-aware gating (per-modality auxiliary heads feed prediction
  entropy into the gate), MixUp, R-Drop, supervised contrastive loss, and
  alpha entropy/diversity regularisation on top of the shared backbone.
- **`StaticFusionModelV2`** — an architecturally identical baseline with the
  fusion weight fixed at α = 0.5, used for a fair ablation against the
  learned gate.

## Architecture

```
image, text
   │
   ▼
CLIP ViT-L/14  (last 4 transformer layers of both encoders fine-tuned)
   │
   ▼
L2-normalise ──► two-layer bidirectional cross-modal attention ──► L2-renormalise
   │
   ├─► auxiliary image head ─┐
   ├─► auxiliary text head ──┼─► entropy ──► dynamic gating network ──► α
   │                         │
   ▼                         ▼
zᵥ = α · f_img        z_t = (1 − α) · f_text
   └──────────────┬──────────────┘
                   ▼
        [zᵥ, z_t, α]  →  MLP classifier  →  hateful / not-hateful
```

## Repository layout

```
src/
  config.py          shared defaults, env-var overrides, device selection
  dataset.py         HatefulMemesDatasetV2 (image/text augmentation)
  model.py           AdaptiveFusionModelV3, StaticFusionModelV2, cross-modal attention, gating
  losses.py          FocalLoss, SupConLoss
  augmentation.py    feature-level MixUp, R-Drop KL divergence
  contrastive.py     two-view SupCon wiring
  regularization.py  alpha entropy / diversity regularisation terms
  curriculum.py      CLIP-similarity easy-first curriculum learning
  checkpointing.py   best-AUROC + rolling periodic checkpoints, resume support
  train_utils.py     shared train/eval loop, optimizer, threshold search
  train_dynamic_v3.py  orchestrator — trains AdaptiveFusionModelV3
  train_static_v3.py   orchestrator — trains StaticFusionModelV2 (baseline)
  ensemble.py        Optuna-based logit blending of the two trained models
  significance.py    paired bootstrap CI + McNemar's test (dynamic vs. static)
  cost_profile.py    parameter / latency / FLOPs inference cost report
  clip_features.py   CLIP feature-extraction helpers

colab/
  hateful_memes_v3_pipeline.ipynb   end-to-end Colab/Kaggle T4 notebook
  make_bundle.py     builds the code+data bundle the notebook consumes

results/             training logs, ablations, alpha-distribution plots
archive/              superseded checkpoints and V1/V2 scripts, kept for comparison
```

## Setup

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch          # or the CUDA wheel matching your GPU — see pytorch.org
pip install -r requirements.txt
```

**Windows** — see [WINDOWS.md](WINDOWS.md) for the full walkthrough (venv
activation, CUDA wheel selection, `NUM_WORKERS=0` DataLoader fallback,
PowerShell env-var syntax).

Data: place the [Hateful Memes](https://huggingface.co/datasets/emily49/hateful-memes)
train/validation parquet files under `Data/` (default paths in
`src/config.py`; override with `DATA_DIR`, or `TRAIN_PARQUET`/`VAL_PARQUET`
for a different dataset entirely).

## Training

```bash
cd src
python train_dynamic_v3.py     # proposed dynamic model
python train_static_v3.py      # static α=0.5 baseline
```

All paths and hyperparameters are configurable via environment variables
(see the docstring in `src/config.py`) so the same scripts run unchanged
locally, on Kaggle, or on Colab — e.g.:

```bash
DATA_DIR=/path/to/data CHECKPOINT_DIR=/path/to/checkpoints \
BATCH_SIZE=8 NUM_EPOCHS=20 PATIENCE=4 CURRICULUM=true \
python train_dynamic_v3.py
```

Checkpoints save on every validation AUROC improvement, plus a rolling
periodic checkpoint every `CHECKPOINT_EVERY` epochs with full optimizer/
scheduler state — pass `RESUME=true` to continue an interrupted run.

For a quick end-to-end sanity check without a full run:

```bash
MAX_SAMPLES=64 NUM_EPOCHS=2 CHECKPOINT_EVERY=1 python train_dynamic_v3.py
```

## Evaluation

```bash
python significance.py   # paired bootstrap CI + McNemar's test, dynamic vs. static
python cost_profile.py   # parameter delta, per-sample latency, FLOPs estimate
python ensemble.py       # Optuna-tuned logit blend of both trained models
```

`significance.py` and `cost_profile.py` both require the two best
checkpoints (`best_model_dynamic_v3.pt`, `best_model_static_v3.pt`) to exist
in `CHECKPOINT_DIR`.

## Cloud training (Kaggle / Colab)

`colab/hateful_memes_v3_pipeline.ipynb` runs the complete pipeline — both
training runs, significance testing, cost profiling, and the optional
ensemble — on a free T4 GPU. Build the bundle it expects with:

```bash
python colab/make_bundle.py
```

then upload `colab/hateful_memes_colab_bundle.zip` to Google Drive (or as a
Kaggle Dataset) and open the notebook.

## Dataset

[Facebook AI Hateful Memes Challenge](https://arxiv.org/abs/2005.04790)
(Kiela et al., NeurIPS 2020), via the
[`emily49/hateful-memes`](https://huggingface.co/datasets/emily49/hateful-memes)
Hugging Face mirror — 10,000 memes (8,500 train / 500 dev / 1,000 test),
adversarially curated so neither the image nor the text alone is sufficient
for classification.
