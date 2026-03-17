"""
setup_env.py — Python virtual environment setup for AMS 520 TP2 Project

Usage:
    python setup_env.py

Works on Mac, Windows, Linux. Python 3.10+ required.
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
VENV_DIR    = PROJECT_DIR / ".venv"
KERNEL_NAME = "tp2"

PACKAGES = {
    "requests"      : "requests",       # Deribit API calls
    "pandas"        : "pandas",         # DataFrames
    "numpy"         : "numpy",          # Math
    "matplotlib"    : "matplotlib",     # Plots
    "seaborn"       : "seaborn",        # Heatmaps
    "jupyter"       : "jupyter",        # Notebooks
    "ipykernel"     : "ipykernel",      # Jupyter kernel
    "tqdm"          : "tqdm",           # Progress bars
    "python-dotenv" : "python-dotenv",  # .env API keys
    "pyarrow"       : "pyarrow",        # Parquet export
}

# ── Colors ─────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"
C = "\033[96m"; R = "\033[91m"; X = "\033[0m"

def ok(msg):    print(f"{G}  ✓ {msg}{X}")
def warn(msg):  print(f"{Y}  ⚠ {msg}{X}")
def err(msg):   print(f"{R}  ✗ {msg}{X}")
def step(n, t): print(f"\n{Y}[{n}/6] {t}{X}")

def venv_bin(name):
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / f"{name}.exe")
    return str(VENV_DIR / "bin" / name)

def run_pip(*args):
    try:
        subprocess.run(
            [venv_bin("pip"), "install", "--upgrade", "--quiet", *args],
            check=True, capture_output=True, text=True
        )
        return True
    except Exception:
        return False

def get_version(pkg):
    try:
        r = subprocess.run(
            [venv_bin("python"), "-c",
             f"import importlib.metadata; print(importlib.metadata.version('{pkg}'))"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else "?"
    except Exception:
        return "?"

# ── Main ──────────────────────────────────────────────────────
print(f"\n{C}{'='*54}{X}")
print(f"{C}  AMS 520 · TP2 Project — Environment Setup      {X}")
print(f"{C}{'='*54}{X}")

# Step 1 — Python version
step(1, "Checking Python version...")
major, minor, micro = sys.version_info[:3]
if (major, minor) < (3, 10):
    err(f"Need Python 3.10+. You have {major}.{minor}. Get it at python.org")
    sys.exit(1)
ok(f"Python {major}.{minor}.{micro}")

# Step 2 — Virtual environment
step(2, "Creating virtual environment (.venv/)...")
if VENV_DIR.exists():
    import shutil
    print("  Removing old .venv...")
    shutil.rmtree(VENV_DIR)
try:
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)],
                   check=True, capture_output=True)
    ok("Virtual environment created")
except subprocess.CalledProcessError as e:
    err(f"Failed to create venv: {e}")
    sys.exit(1)

# Step 3 — Upgrade pip
step(3, "Upgrading pip, setuptools, wheel...")
if run_pip("pip", "setuptools", "wheel"):
    ok("pip upgraded")
else:
    warn("pip upgrade skipped — continuing anyway")

# Step 4 — Install packages
step(4, "Installing packages...")
success, failed = [], []
for name, install in PACKAGES.items():
    print(f"  {name:<22}", end=" ", flush=True)
    if run_pip(install):
        v = get_version(name.replace("-", "_"))
        print(f"{G}✓  {v}{X}")
        success.append(name)
    else:
        print(f"{R}FAILED{X}")
        failed.append(name)

print(f"\n  {G}{len(success)} installed{X}  |  {R if failed else G}{len(failed)} failed{X}")
if failed:
    warn("Retry manually after activating:")
    warn(f"  pip install {' '.join(failed)}")

# Step 5 — Jupyter kernel
step(5, f"Registering Jupyter kernel '{KERNEL_NAME}'...")
try:
    subprocess.run([
        venv_bin("python"), "-m", "ipykernel", "install", "--user",
        f"--name={KERNEL_NAME}",
        f"--display-name=TP2 Project (Python {major}.{minor})"
    ], check=True, capture_output=True)
    ok(f"Kernel: 'TP2 Project (Python {major}.{minor})'")
except Exception:
    warn("Kernel registration failed — select .venv manually in Jupyter")

# Step 6 — Project structure + .env
step(6, "Creating directories and .env template...")
for d in ["data/db", "data/raw", "data/processed", "logs"]:
    (PROJECT_DIR / d).mkdir(parents=True, exist_ok=True)
ok("Directories created")

env_path = PROJECT_DIR / ".env"
if not env_path.exists():
    env_path.write_text(
        "# AMS 520 TP2 Project — API Keys\n\n"
        "# Deribit public API — no key needed for market data\n"
        "DERIBIT_API_KEY=\n\n"
        "# Tardis.dev — needed for 90-day historical archive\n"
        "# Sign up free at https://tardis.dev to get a key\n"
        "TARDIS_API_KEY=your_tardis_key_here\n"
    )
    ok(".env created — add your Tardis key when ready")
else:
    ok(".env already exists")

# ── Done ──────────────────────────────────────────────────────
ACT = r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
print(f"""
{C}{'='*54}{X}
{G}  ✓ Setup complete!{X}
{C}{'='*54}{X}

  {Y}1. Activate the environment:{X}
       {ACT}

  {Y}2. Test one snapshot now:{X}
       python main.py --mode once

  {Y}3. Backfill 30 days of history (run once):{X}
       python main.py --mode history --days 30

  {Y}4. Start live hourly collection:{X}
       python main.py --mode full

  {Y}5. Open Jupyter:{X}
       jupyter notebook
       → kernel: TP2 Project (Python {major}.{minor})

  {Y}6. Deactivate when done:{X}
       deactivate
""")
