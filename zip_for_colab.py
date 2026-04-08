"""
zip_for_colab.py — Zip the tp2_project code for uploading to Google Drive.

Usage:
    python zip_for_colab.py

Output:
    ../tp2_project_code.zip   (one level up, next to the project folder)

What's included:  all .py files, .ipynb, requirements.txt
What's excluded:  data/, checkpoints/, .venv/, __pycache__, logs/, results/,
                  papers/, docs/, *.db, *.npz, *.json, *.pt, *.png
"""

import zipfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_ZIP   = PROJECT_ROOT.parent / "tp2_project_code.zip"

# Directories to skip entirely
SKIP_DIRS = {
    ".venv", "__pycache__", ".git", ".claude",
    "data", "checkpoints", "logs", "results", "papers", "docs",
}

# File extensions to skip
SKIP_EXTS = {
    ".db", ".npz", ".npy", ".pt", ".pth",
    ".png", ".jpg", ".jpeg", ".pdf",
    ".pyc", ".pyo", ".DS_Store",
}

# Specific files to skip
SKIP_FILES = {"zip_for_colab.py", ".env", ".DS_Store"}


def should_include(path: Path) -> bool:
    if path.name in SKIP_FILES:
        return False
    if path.suffix in SKIP_EXTS:
        return False
    # Skip large json files (metadata)
    if path.suffix == ".json" and path.stat().st_size > 1_000_000:
        return False
    return True


def build_zip():
    included = []
    skipped  = []

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)

            # Prune skip-dirs in-place so os.walk doesn't descend into them
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in sorted(files):
                fpath = root_path / fname
                if not should_include(fpath):
                    skipped.append(fpath.relative_to(PROJECT_ROOT))
                    continue

                arcname = fpath.relative_to(PROJECT_ROOT)
                zf.write(fpath, arcname)
                included.append(arcname)

    size_kb = OUTPUT_ZIP.stat().st_size / 1024
    print(f"Output : {OUTPUT_ZIP}")
    print(f"Size   : {size_kb:.0f} KB")
    print(f"\nIncluded ({len(included)} files):")
    for f in included:
        print(f"  {f}")
    print(f"\nSkipped ({len(skipped)} files — data/checkpoints/etc)")


if __name__ == "__main__":
    build_zip()
