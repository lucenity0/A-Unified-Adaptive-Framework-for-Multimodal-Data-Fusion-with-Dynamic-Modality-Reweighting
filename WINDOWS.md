# Running the pipeline on Windows

The code itself is cross-platform — device selection automatically picks
CUDA on Windows (the `mps` check is a no-op there), all paths go through
`os.path` / env vars, and every script guards its entry point for Windows'
spawn-based multiprocessing. This guide covers setup and the handful of
Windows-specific knobs.

## 1. Environment setup (once)

From the repo root, in **PowerShell**:

```powershell
py -3.11 -m venv venv          # any Python 3.10+ works
venv\Scripts\Activate.ps1      # (cmd.exe: venv\Scripts\activate.bat)

# torch first, matched to your hardware — check https://pytorch.org for the
# current CUDA index URL if this one is outdated:
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU
# pip install torch                                                    # CPU only

pip install -r requirements.txt
```

Verify the GPU is seen:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 2. Configure via environment variables

Same variables as Colab/Kaggle (see `src/config.py` for the full list).
PowerShell syntax:

```powershell
$env:DATA_DIR        = "C:\path\to\Hateful memes static copy\Data"
$env:CHECKPOINT_DIR  = "C:\path\to\checkpoints"
$env:RESULTS_DIR     = "C:\path\to\results"
$env:BATCH_SIZE      = "8"        # lower to 4 (+ GRAD_ACCUM=8) if CUDA OOMs
$env:PYTHONUTF8      = "1"        # so the ✅/💾 progress glyphs print cleanly
```

(cmd.exe uses `set DATA_DIR=C:\...` instead; unset with `Remove-Item Env:DATA_DIR`.)

If you run from inside `src\`, the defaults (`..\Data`, `..\checkpoints`)
already resolve correctly and you only need `DATA_DIR` when the data lives
elsewhere.

## 3. Run

```powershell
cd src
python train_dynamic_v3.py     # dynamic model
python train_static_v3.py      # static baseline
python significance.py         # needs both best checkpoints
python cost_profile.py         # inference cost report
```

Smoke test first (a few minutes, verifies the full path works):

```powershell
$env:MAX_SAMPLES = "64"; $env:NUM_EPOCHS = "2"; $env:CHECKPOINT_EVERY = "1"
python train_dynamic_v3.py
Remove-Item Env:MAX_SAMPLES, Env:NUM_EPOCHS, Env:CHECKPOINT_EVERY   # before real runs
```

## 4. Windows-specific notes

- **DataLoader hangs or is very slow at epoch start** → set
  `$env:NUM_WORKERS = "0"`. Windows uses spawn-based multiprocessing, which
  re-imports modules and pickles the CLIP processor per worker; 0 disables
  worker processes entirely and is the reliable fallback.
- **Resuming after an interruption** → `$env:RESUME = "true"` and rerun the
  same training command; it continues from the rolling `*_last.pt`
  checkpoint (saved every `CHECKPOINT_EVERY` epochs).
- **Rebuilding the Colab bundle** → `python colab\make_bundle.py`
  (cross-platform; `make_bundle.sh` needs bash).
- **Console encoding errors** (`UnicodeEncodeError` on ✅/💾) → you skipped
  `PYTHONUTF8=1` above; set it, or use Windows Terminal which defaults to UTF-8.
