#!/bin/bash
# =============================================================
#  AMS 520 — TP2 Project Environment Setup
#  Run this ONCE when you first clone/unzip the project.
#
#  Usage:
#    chmod +x setup_env.sh
#    ./setup_env.sh
#
#  What it does:
#    1. Creates a Python virtual environment (.venv/)
#    2. Installs all dependencies from requirements.txt
#    3. Registers a Jupyter kernel named "tp2"
#    4. Prints how to activate and run
# =============================================================

set -e   # stop on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="python3"
KERNEL_NAME="tp2"

# ── Colors ────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  AMS 520 · TP2 Project — Environment Setup   ${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# ── Check Python version ───────────────────────────────────────
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo -e "${RED}ERROR: Python 3.10+ required. Found: $PY_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Python $PY_VERSION${NC}"

# ── Create virtual environment ─────────────────────────────────
echo ""
echo -e "${YELLOW}[2/5] Creating virtual environment at .venv/ ...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "  Existing .venv found — removing and recreating..."
    rm -rf "$VENV_DIR"
fi

$PYTHON -m venv "$VENV_DIR"
echo -e "${GREEN}  ✓ Virtual environment created${NC}"

# ── Activate venv ──────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Upgrade pip ────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/5] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}  ✓ pip upgraded to $(pip --version | awk '{print $2}')${NC}"

# ── Install dependencies ───────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/5] Installing dependencies from requirements.txt...${NC}"
echo ""

pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo -e "${GREEN}  ✓ All dependencies installed${NC}"

# ── Register Jupyter kernel ────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/5] Registering Jupyter kernel '${KERNEL_NAME}'...${NC}"

python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="TP2 Project (Python $PY_VERSION)"
echo -e "${GREEN}  ✓ Kernel registered as '${KERNEL_NAME}'${NC}"

# ── Create data directories ────────────────────────────────────
mkdir -p "$PROJECT_DIR/data/db"
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/processed"
mkdir -p "$PROJECT_DIR/logs"

# ── Print installed versions ───────────────────────────────────
echo ""
echo -e "${CYAN}Installed package versions:${NC}"
pip show requests pandas numpy matplotlib seaborn jupyter 2>/dev/null | \
    grep -E "^(Name|Version):" | \
    paste - - | \
    awk '{printf "  %-20s %s\n", $2, $4}'

# ── Done ───────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN}  ✓ Setup complete!${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo -e "  ${YELLOW}To activate the environment:${NC}"
echo -e "    source .venv/bin/activate"
echo ""
echo -e "  ${YELLOW}To run the project (terminal):${NC}"
echo -e "    source .venv/bin/activate"
echo -e "    python main.py --mode once          # test one snapshot"
echo -e "    python main.py --mode history       # backfill 30 days"
echo -e "    python main.py --mode full          # history + live"
echo ""
echo -e "  ${YELLOW}To open Jupyter notebook:${NC}"
echo -e "    source .venv/bin/activate"
echo -e "    jupyter notebook"
echo -e "    → Select kernel: 'TP2 Project (Python $PY_VERSION)'"
echo ""
echo -e "  ${YELLOW}To deactivate when done:${NC}"
echo -e "    deactivate"
echo ""
