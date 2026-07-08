"""
Build hateful_memes_colab_bundle.zip for the Colab/Kaggle notebook.
Cross-platform replacement for make_bundle.sh (works on Windows/macOS/Linux).

Contents: src/*.py + Data/*.parquet (code + dataset, no checkpoints).

Usage (from the repo root):
    python colab/make_bundle.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "colab" / "hateful_memes_colab_bundle.zip"


def main():
    files = sorted(ROOT.glob("src/*.py")) + sorted(ROOT.glob("Data/*.parquet"))
    if not any(f.suffix == ".parquet" for f in files):
        raise SystemExit("No parquet files found under Data/ — run from the repo root.")

    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f.relative_to(ROOT).as_posix()
            print(f"  adding {arcname}")
            zf.write(f, arcname)

    print(f"\nBuilt {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
