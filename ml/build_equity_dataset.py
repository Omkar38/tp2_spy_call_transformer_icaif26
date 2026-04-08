"""
ml/build_equity_dataset.py — Fast vectorized dataset builder for equity (SPX) data.

The standard ml/features.py does per-sample SQL queries — fine for crypto
(~thousands of samples), but too slow for equity (320K samples × 487 days).

This builder pre-fetches everything into RAM in bulk, then assembles tensors
entirely in numpy without any per-sample SQL calls.

Runtime: ~5 minutes vs ~hours for the generic builder.

Usage:
    python -m ml.build_equity_dataset
    # writes data/processed/equity_dataset.npz + equity_metadata.json

Then train:
    python -m ml.train_equity
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("equity_dataset")

# ── Feature dims (same as ml/features.py) ─────────────────────
D_TEMPORAL = 19
D_SURFACE  = 8
SEQ_LEN    = 60    # trading days lookback
MAX_INSTR  = 300   # max instruments per snapshot


def build_equity_dataset(db_path: str = None) -> dict:
    """
    Build the complete SPX equity ML dataset.

    Strategy:
    1. Load ALL violation + label data in one query
    2. Load ALL surface data (all hours) into a dict keyed by (hour, expiry, strike)
    3. Load regime cache (already done in features.py style, but bulk)
    4. Assemble tensors WITHOUT per-sample SQL queries

    Returns same format as ml/features.build_dataset().
    """
    if db_path is None:
        db_path = str(cfg.DB_PATH)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    log.info(f"Building equity dataset from {db_path}")

    # ── 1. Load labeled violations ───────────────────────────
    log.info("Loading labeled violations...")
    rows = conn.execute("""
        SELECT
            v.id, v.snapshot_hour,
            v.T1, v.T2, v.K1, v.K2,
            v.C_K1_T1, v.C_K2_T1, v.C_K1_T2, v.C_K2_T2,
            v.btc_spot, v.violation_pct, v.net_edge, v.avg_spread_pct,
            v.maturity_gap, v.strike_gap_pct,
            v.moneyness_K1, v.moneyness_K2,
            r.revert_speed
        FROM violations v
        JOIN (
            SELECT violation_id,
                   revert_speed,
                   MIN(CASE revert_speed
                       WHEN 'superfast' THEN 0 WHEN 'fast' THEN 1
                       WHEN 'medium' THEN 2 WHEN 'slow' THEN 3 WHEN 'never' THEN 4
                       ELSE 99 END) AS speed_rank
            FROM reversions
            WHERE revert_speed IN ('superfast','fast','medium','slow','never')
            GROUP BY violation_id
        ) r ON r.violation_id = v.id
        ORDER BY v.snapshot_hour ASC
    """).fetchall()

    N = len(rows)
    log.info(f"Labeled violations: {N:,}")
    n_pos = sum(1 for r in rows if r["revert_speed"] in ("superfast", "fast"))
    log.info(f"  Fast (label=1): {n_pos:,} ({100*n_pos/max(N,1):.2f}%)")

    # ── 2. Load ALL surface data into memory ─────────────────
    log.info("Loading full surface into memory (all hours × all instruments)...")
    # surface_data[hour][(expiry, strike)] = {mid, spread_pct, moneyness, tte, mark_iv, oi, vol}
    surf_rows = conn.execute("""
        SELECT snapshot_hour, expiry, strike, option_type,
               mid_price, spread_pct, moneyness, time_to_expiry_d,
               mark_iv, open_interest, volume_24h
        FROM snapshots
        WHERE is_clean = 1 AND option_type = 'C' AND mid_price > 0
        ORDER BY snapshot_hour, expiry, strike
    """).fetchall()

    surface_data = defaultdict(dict)  # hour → {(expiry, strike) → tuple}
    for r in surf_rows:
        surface_data[r["snapshot_hour"]][(r["expiry"], r["strike"])] = (
            r["mid_price"],
            r["spread_pct"] or 0.0,
            r["moneyness"] or 0.0,
            r["time_to_expiry_d"] or 0.0,
            (r["mark_iv"] or 0.0) / 100.0,
            math.log1p(r["open_interest"] or 0),
            math.log1p(r["volume_24h"] or 0),
        )
    log.info(f"Surface loaded: {len(surface_data)} hours, {len(surf_rows):,} instruments total")

    # ── 3. Build hourly regime cache ─────────────────────────
    log.info("Building regime cache...")
    all_hours = sorted(surface_data.keys())
    hour_idx = {h: i for i, h in enumerate(all_hours)}

    # Spot per hour
    spot_rows = conn.execute("""
        SELECT snapshot_hour, AVG(underlying_price) as spot, COUNT(*) as n_clean
        FROM snapshots WHERE is_clean=1
        GROUP BY snapshot_hour ORDER BY snapshot_hour
    """).fetchall()
    spot_map = {r["snapshot_hour"]: (r["spot"] or 0.0, r["n_clean"]) for r in spot_rows}

    # IV per hour
    iv_rows = conn.execute("""
        SELECT snapshot_hour, AVG(mark_iv) as iv
        FROM snapshots WHERE is_clean=1 AND mark_iv IS NOT NULL
        GROUP BY snapshot_hour
    """).fetchall()
    iv_map = {r["snapshot_hour"]: r["iv"] or 0.0 for r in iv_rows}

    # Violation stats per hour
    viol_rows = conn.execute("""
        SELECT snapshot_hour, COUNT(*) as n, AVG(violation_pct) as avg, MAX(violation_pct) as mx
        FROM violations GROUP BY snapshot_hour
    """).fetchall()
    viol_map = {r["snapshot_hour"]: (r["n"], r["avg"] or 0.0, r["mx"] or 0.0) for r in viol_rows}

    # Regime features per hour
    regime_cache = {}
    prev_spot = None
    for h in all_hours:
        spot, n_clean = spot_map.get(h, (0.0, 0))
        spot_lr = math.log(spot / prev_spot) if spot and prev_spot and prev_spot > 0 else 0.0
        n_v, avg_v, max_v = viol_map.get(h, (0, 0.0, 0.0))
        iv = iv_map.get(h, 0.0)
        dt = datetime.fromisoformat(h)
        regime_cache[h] = {
            "spot": spot or 0.0,
            "spot_log_return": spot_lr,
            "iv_level": iv,
            "iv_skew": 0.0,     # NULL in equity ohlcv — leave 0
            "term_slope": 0.0,  # NULL in equity ohlcv — leave 0
            "realized_vol": 0.0,
            "n_clean": n_clean,
            "n_violations": n_v,
            "avg_viol_pct": avg_v,
            "max_viol_pct": max_v,
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),
        }
        prev_spot = spot or prev_spot

    log.info(f"Regime cache: {len(regime_cache)} hours")

    # ── 4. Assemble tensors ──────────────────────────────────
    log.info(f"Assembling tensors: {N:,} samples × (seq={SEQ_LEN}, d_t={D_TEMPORAL}) and (max_instr={MAX_INSTR}, d_s={D_SURFACE})...")

    temporal     = np.zeros((N, SEQ_LEN, D_TEMPORAL), dtype=np.float32)
    surface      = np.zeros((N, MAX_INSTR, D_SURFACE),  dtype=np.float32)
    surface_mask = np.zeros((N, MAX_INSTR),              dtype=np.float32)
    labels       = np.zeros(N,                           dtype=np.int64)
    metadata     = []

    for i, row in enumerate(rows):
        if i % 10000 == 0:
            log.info(f"  {i}/{N}...")

        det_hour = row["snapshot_hour"]
        spot     = row["btc_spot"] or 1.0
        t1, t2   = row["T1"], row["T2"]
        k1, k2   = row["K1"], row["K2"]
        speed    = row["revert_speed"]
        label    = 1 if speed in ("superfast", "fast") else 0

        labels[i] = label
        metadata.append({
            "id": row["id"],
            "snapshot_hour": det_hour,
            "T1": t1, "T2": t2, "K1": k1, "K2": k2,
            "revert_speed": speed, "label": label,
        })

        # ── Violation-level static features (broadcast across all timesteps) ──
        # These replaced 5 dead features (realized_vol, iv_level, iv_skew,
        # term_slope, quad_avg_spread — all zero for equity ohlcv-1d).
        # Analysis showed these have clear separability between fast vs slow:
        #   maturity_gap:  fast=229d vs medium=298d
        #   strike_gap:    fast=0.039 vs medium=0.078
        #   moneyness_K1:  fast=1.013 vs medium=0.930
        mat_gap_norm  = (row["maturity_gap"] or 0) / 500.0          # ~0–4, typ 0.3–0.6
        strike_gap    = row["strike_gap_pct"] or 0.0                 # ~0–1.6, typ 0.03–0.08
        mono_k1       = row["moneyness_K1"] or 1.0                   # ~0.5–3.0
        mono_k2       = row["moneyness_K2"] or 1.0
        raw_viol      = min(row["violation_pct"] or 0, 200.0)
        log_viol_pct  = math.log1p(raw_viol) / math.log1p(200.0)    # normalized [0,1]

        # ── Temporal stream ──────────────────────────────
        det_idx = hour_idx.get(det_hour, -1)
        if det_idx >= 0:
            start_idx = max(0, det_idx - SEQ_LEN + 1)
            window_hours = all_hours[start_idx: det_idx + 1]
            offset = SEQ_LEN - len(window_hours)

            for j, h in enumerate(window_hours):
                ri = offset + j
                reg = regime_cache.get(h, {})

                temporal[i, ri, 0] = reg.get("spot_log_return", 0.0)
                temporal[i, ri, 1] = mat_gap_norm   # was realized_vol (always 0)
                temporal[i, ri, 2] = mono_k1         # was iv_level (always 0)
                temporal[i, ri, 3] = mono_k2         # was iv_skew (always 0)
                temporal[i, ri, 4] = strike_gap      # was term_slope (always 0)
                temporal[i, ri, 5] = reg.get("n_clean", 0) / 500.0
                temporal[i, ri, 6] = reg.get("n_violations", 0) / 1000.0
                temporal[i, ri, 7] = reg.get("avg_viol_pct", 0.0)
                temporal[i, ri, 8] = reg.get("max_viol_pct", 0.0)

                # Quad prices from in-memory surface
                surf_h = surface_data.get(h, {})
                c11 = surf_h.get((t1, k1), (0,))[0]
                c21 = surf_h.get((t1, k2), (0,))[0]
                c12 = surf_h.get((t2, k1), (0,))[0]
                c22 = surf_h.get((t2, k2), (0,))[0]

                temporal[i, ri, 9]  = c11 / max(spot, 1)
                temporal[i, ri, 10] = c21 / max(spot, 1)
                temporal[i, ri, 11] = c12 / max(spot, 1)
                temporal[i, ri, 12] = c22 / max(spot, 1)

                lhs = c11 * c22
                rhs = c12 * c21
                temporal[i, ri, 13] = (rhs / lhs) if lhs > 0 else 1.0
                temporal[i, ri, 14] = log_viol_pct   # was quad_avg_spread (always 0)

                hod = reg.get("hour_of_day", 0)
                dow = reg.get("day_of_week", 0)
                temporal[i, ri, 15] = math.sin(2 * math.pi * hod / 24)
                temporal[i, ri, 16] = math.cos(2 * math.pi * hod / 24)
                temporal[i, ri, 17] = math.sin(2 * math.pi * dow / 7)
                temporal[i, ri, 18] = math.cos(2 * math.pi * dow / 7)

        # ── Surface stream ────────────────────────────────
        violation_legs = {(t1, k1), (t1, k2), (t2, k1), (t2, k2)}
        surf_det = surface_data.get(det_hour, {})
        instr_list = sorted(surf_det.keys())[:MAX_INSTR]
        n_actual = len(instr_list)
        surface_mask[i, :n_actual] = 1.0

        for si, (expiry, strike) in enumerate(instr_list):
            mid, sprd, mono, tte, iv, log_oi, log_vol = surf_det[(expiry, strike)]
            surface[i, si, 0] = mono
            surface[i, si, 1] = math.log1p(max(tte, 0))
            surface[i, si, 2] = iv
            surface[i, si, 3] = sprd
            surface[i, si, 4] = mid / max(spot, 1)
            surface[i, si, 5] = log_oi
            surface[i, si, 6] = log_vol
            surface[i, si, 7] = 1.0 if (expiry, strike) in violation_legs else 0.0

    pos_rate = labels.sum() / max(len(labels), 1) * 100
    log.info(f"Dataset complete: {N:,} samples | positive rate {pos_rate:.2f}%")
    log.info(f"Temporal: {temporal.shape} | Surface: {surface.shape}")

    conn.close()

    return {
        "temporal":     temporal,
        "surface":      surface,
        "surface_mask": surface_mask,
        "labels":       labels,
        "metadata":     metadata,
    }


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format=cfg.LOG_FORMAT)

    os.makedirs("data/processed", exist_ok=True)

    dataset_dict = build_equity_dataset()

    # Save compressed
    np.savez_compressed(
        "data/processed/equity_dataset.npz",
        temporal=dataset_dict["temporal"],
        surface=dataset_dict["surface"],
        surface_mask=dataset_dict["surface_mask"],
        labels=dataset_dict["labels"],
    )
    with open("data/processed/equity_metadata.json", "w") as f:
        json.dump(dataset_dict["metadata"], f)

    log.info("Saved: data/processed/equity_dataset.npz")
    log.info("Saved: data/processed/equity_metadata.json")
    print(f"\nDataset: {len(dataset_dict['labels']):,} samples saved.")
