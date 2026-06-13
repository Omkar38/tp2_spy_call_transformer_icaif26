#!/usr/bin/env bash
# Turnkey runner for download_spy_options.py
# Usage:
#   ./run_download.sh                 # full history (2013-04 -> today), default 2-min EOD window
#   ./run_download.sh --dry-run       # print ranges / record counts / cost, download nothing
#   ./run_download.sh --start 2024-01-01 --window-min 1   # pass any script flags through
set -euo pipefail
cd "$(dirname "$0")"

# API key from API_key.txt (whitespace-stripped)
export DATABENTO_API_KEY="$(tr -d '[:space:]' < API_key.txt)"

# First-time setup (uncomment once, or run manually):
#   python3 -m pip install -r requirements.txt

exec python3 download_spy_options.py --out ./spy_options_daily "$@"
