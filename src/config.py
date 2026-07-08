"""
config.py
=========
Shared configuration for the V3 training pipelines.

Responsibilities
----------------
  • base_config()          — defaults common to both train scripts
                             (data paths, CLIP model, optimiser, focal loss).
  • apply_env_overrides()  — let Colab/Kaggle/shell runs override any simple
                             config value through environment variables, so
                             the same scripts run locally and in the cloud
                             without edits.
  • get_device()           — mps → cuda → cpu preference. (mps is checked
                             first but does not exist on Linux runtimes, so
                             Colab/Kaggle fall through to cuda.)

Path logic is dataset-agnostic: nothing here assumes Hateful Memes. The
default parquet filenames are just defaults — point DATA_DIR at another
dataset's directory and/or set TRAIN_PARQUET / VAL_PARQUET to full paths
(e.g. for CrisisMMD) and everything downstream follows.

Environment variables
---------------------
  DATA_DIR          directory containing the parquet files
  TRAIN_PARQUET     full path to the training parquet (wins over DATA_DIR)
  VAL_PARQUET       full path to the validation parquet (wins over DATA_DIR)
  CHECKPOINT_DIR    where checkpoints are written
  CHECKPOINT_NAME   filename of the best-model checkpoint
  CLIP_MODEL        HF model id, e.g. openai/clip-vit-large-patch14
  BATCH_SIZE, GRAD_ACCUM, NUM_EPOCHS, PATIENCE          ints
  NUM_WORKERS       DataLoader workers (set 0 on Windows if loading hangs)
  CHECKPOINT_EVERY  periodic-checkpoint interval in epochs (0 disables)
  RESUME            true/false — resume from the rolling periodic checkpoint
  CURRICULUM        true/false
  CURRICULUM_EASY_THRESH   float
  CURRICULUM_PHASE1_END    int
"""

import os

import torch

# Default filenames looked up inside DATA_DIR when TRAIN_PARQUET / VAL_PARQUET
# are not set explicitly. Override per dataset.
DEFAULT_TRAIN_FILE = "train-00000-of-00001-6587b3a58d350036.parquet"
DEFAULT_VAL_FILE   = "validation-00000-of-00001-1508d9e5032c2c1f.parquet"


def _as_bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "on"}


# env var → (config key, caster)
ENV_MAP = {
    "TRAIN_PARQUET":          ("train_parquet",          str),
    "VAL_PARQUET":            ("val_parquet",            str),
    "CHECKPOINT_DIR":         ("checkpoint_dir",         str),
    "CHECKPOINT_NAME":        ("checkpoint_name",        str),
    "CLIP_MODEL":             ("clip_model",             str),
    "BATCH_SIZE":             ("batch_size",             int),
    "GRAD_ACCUM":             ("grad_accum",             int),
    "NUM_WORKERS":            ("num_workers",            int),
    "NUM_EPOCHS":             ("num_epochs",             int),
    "PATIENCE":               ("patience",               int),
    "CHECKPOINT_EVERY":       ("checkpoint_every",       int),
    "MAX_SAMPLES":            ("max_samples",            int),
    "RESUME":                 ("resume",                 _as_bool),
    "CURRICULUM":             ("curriculum",             _as_bool),
    "CURRICULUM_EASY_THRESH": ("curriculum_easy_thresh", float),
    "CURRICULUM_PHASE1_END":  ("curriculum_phase1_end",  int),
}


def base_config() -> dict:
    """Config values shared by the dynamic and static V3 training scripts."""
    data_dir = os.environ.get("DATA_DIR", "../Data")

    return {
        # ── data ──────────────────────────────────────────────────────────────
        "train_parquet":  os.path.join(data_dir, DEFAULT_TRAIN_FILE),
        "val_parquet":    os.path.join(data_dir, DEFAULT_VAL_FILE),

        # ── model ─────────────────────────────────────────────────────────────
        "clip_model":      "openai/clip-vit-large-patch14",
        "unfreeze_layers": 4,
        "attn_dropout":    0.2,

        # ── training ──────────────────────────────────────────────────────────
        "batch_size":      8,
        "grad_accum":      4,           # effective batch = 32
        "num_workers":     2,           # DataLoader workers; set 0 on Windows
                                        # if loading hangs (spawn-based mp)
        "max_samples":     0,           # subsample datasets (smoke tests); 0 = all
        "num_epochs":      20,
        "patience":        4,   # tightened from 7 — dynamic model peaks early
                                # (V2 peaked at epoch 3); 4 not 3 so the
                                # curriculum phase-1→2 transition isn't cut off

        # ── optimiser ─────────────────────────────────────────────────────────
        "clip_lr":         2e-6,        # lower LR for fine-tuning CLIP layers
        "fusion_lr":       1e-4,
        "weight_decay":    0.01,
        "warmup_ratio":    0.1,

        # ── focal loss ────────────────────────────────────────────────────────
        "focal_alpha":     0.75,
        "focal_gamma":     2.0,
        "label_smoothing": 0.1,

        # ── checkpointing ─────────────────────────────────────────────────────
        "checkpoint_dir":   "../checkpoints",
        "checkpoint_every": 2,          # periodic save every N epochs; 0 = off
        "resume":           False,      # resume from rolling periodic checkpoint
    }


def apply_env_overrides(cfg: dict) -> dict:
    """Override cfg values from environment variables (see ENV_MAP)."""
    for env_name, (cfg_name, caster) in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value is not None and value != "":
            cfg[cfg_name] = caster(value)
    return cfg


def get_device() -> torch.device:
    return torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
