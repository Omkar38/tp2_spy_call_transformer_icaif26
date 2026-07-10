"""
Legacy TP2-MV-ALiBi transformer runner.

This wrapper preserves the pre-fix behavior from the earlier GPU run, where
datetime64[ms] values were converted as if they were nanoseconds. That caused
the historical-attention lookup to find no usable prior TP2 tokens in the
uploaded Colab run. We keep this file only for controlled comparison against
the corrected historical-attention runner:

    corrected: TP2MVTransformer/03_train_mv_alibi_transformer.py
    legacy:    TP2MVTransformer/03_train_mv_alibi_transformer_legacy_no_history.py

Do not use the legacy runner as the main publishable transformer. Its purpose is
to reproduce the earlier result and quantify whether true history helps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
BASE_RUNNER = THIS_FILE.with_name("03_train_mv_alibi_transformer.py")

spec = importlib.util.spec_from_file_location("tp2_mv_alibi_corrected", BASE_RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def legacy_date_to_day_number(values: pd.Series):
    """Reproduce the old datetime64[ms] conversion behavior."""
    return pd.to_datetime(values).astype("int64").to_numpy() // 86_400_000_000_000


runner.date_to_day_number = legacy_date_to_day_number

if __name__ == "__main__":
    print("=" * 72)
    print("RUNNING LEGACY NO-HISTORY-COMPATIBLE TRANSFORMER")
    print("This preserves the earlier datetime conversion behavior for comparison.")
    print("Use 03_train_mv_alibi_transformer.py for the corrected main run.")
    print("=" * 72)
    runner.main()
