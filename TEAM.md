# TP2 Project — Team Reference Guide

**Course:** AMS 520 — Machine Learning in Quantitative Finance
**Team:** Omkar (pipeline lead), Indu (Deribit/crypto data), Jyotir (transfer learning research)
**Advisor:** Prof. Polak

---

## What This Project Does

We replicate Glasserman & Pirjol (2026) *"When Are Option Prices TP₂?"* on SPX equity options, then train a transformer to predict which violations revert fastest.

**TP₂ (Total Positivity of order 2):** Call option prices must satisfy:

```
C(K1,T1) · C(K2,T2) ≥ C(K2,T1) · C(K1,T2)
```

Equivalently, the ratio `C(K2,T)/C(K1,T)` must be non-decreasing in T. When it isn't, that's a violation — a potential arbitrage opportunity.

**Paper replication result (confirmed):**
- 321,903 violations over 487 trading days (~661/day)
- 98.7% revert within 20 trading days
- Only 1.33% revert within 2 trading days ("fast") — this is our ML target

---

## Repository Structure

```
tp2_project/
├── config.py                   # Single source of truth — all thresholds, paths, toggles
│
├── collector/
│   ├── deribit.py              # Live BTC/ETH/SOL options from Deribit API
│   ├── databento.py            # SPX equity options via Databento OPRA
│   ├── spx_spot.py             # Backfill SPX spot price via yfinance (^GSPC)
│   ├── historical.py / v2      # Historical data collectors
│   └── live.py / v2            # Live collection loops
│
├── processor/
│   ├── cleaner.py              # Filter bad rows (spread, OI, moneyness, IV bounds)
│   └── features.py             # Compute moneyness, TTM, mark IV, etc.
│
├── detector/
│   └── tp2.py                  # Detect TP2 violations, label reversions
│
├── storage/
│   └── db.py                   # SQLite schema + write helpers
│
├── ml/
│   ├── build_equity_dataset.py # Fast vectorized dataset builder for SPX (320K samples)
│   ├── features.py             # Generic feature extractor (for crypto)
│   ├── dataset.py              # PyTorch Dataset, chronological split, balanced sampler
│   ├── transformer.py          # DS-CAT model definition
│   ├── train.py                # Training loop, focal loss, threshold calibration
│   └── backtest.py             # Backtesting utilities
│
├── train.ipynb                 # Main Colab training notebook (run this on T4 GPU)
├── zip_for_colab.py            # Zip code for Drive upload (run locally before Colab)
├── main.py                     # Entry point for live collection
└── checkpoints/equity/
    └── best_model.pt           # Trained DS-CAT checkpoint (SPX, F1=0.287)
```

---

## Asset Class Toggle

Everything switches between equity and crypto via one setting in `config.py`:

```python
ASSET_CLASS = "equity"   # SPX options via Databento
ASSET_CLASS = "crypto"   # BTC/ETH/SOL via Deribit
```

Or via environment variable (used in Colab):
```bash
export TP2_ASSET_CLASS=equity
```

---

## ML Architecture: DS-CAT

**Dual-Stream Cross-Attention Transformer** — two input streams:

```
Temporal (60 days × 19 features) → Linear → PosEnc → TransformerEncoder ─┐
                                                                           ├→ CrossAttn → Pool → MLP → P(fast)
Surface  (300 options × 8 features) → Linear → TransformerEncoder ────────┘
```

**Temporal features (19):** spot log-return, maturity gap, moneyness K1/K2, strike gap, n_clean instruments, n_violations, avg/max violation %, four normalized option prices (C11/C21/C12/C22), TP2 ratio, log-violation magnitude, hour/day cyclical encoding.

**Surface features (8):** moneyness, log-TTM, implied vol, spread %, normalized price, log-OI, log-volume, violation leg flag.

**Cross-attention:** temporal queries surface — "given recent market dynamics, which options are informative for predicting fast reversion?"

**Model size:** 223K parameters — deliberately small to avoid overfitting on ~3K positive training samples.

---

## Training Setup

**Key design choices:**

| Decision | Why |
|---|---|
| Chronological split 70/15/15 | Random shuffle leaks future into training |
| WeightedRandomSampler (20% pos/batch) | 1.33% positives → 1 per 75 negatives per batch without it |
| Focal Loss γ=1.5 | Down-weights easy negatives; raw cross-entropy collapses to all-negative |
| Alpha capped at [0.6, 0.9] | Raw n_neg/n_total = 0.987 → model predicts everything positive |
| No early stopping | Val period (Oct–Dec 2025) is a harder regime; val F1 ≈ 0.12 regardless |
| Threshold calibration on val | Sweep [0.05, 0.95] for best val F1, apply to test |
| CosineAnnealingLR | Smooth single decay; warm restarts caused val loss spikes |

---

## Results So Far

**Paper replication:**

| Metric | Value |
|---|---|
| Trading days | 487 (Apr 2024 – Mar 2026) |
| Total violations | 321,903 (~661/day) |
| Revert ≤ 2 days (fast) | 4,268 (1.33%) |
| Revert ≤ 20 days | 98.7% |
| Never revert | 1.3% |

**DS-CAT (SPX, checkpoint in `checkpoints/equity/best_model.pt`):**

| Metric | Value |
|---|---|
| F1 | 0.287 |
| Precision | 0.233 |
| Recall | 0.373 |
| Accuracy | 0.959 |
| Threshold | 0.73 |
| vs. random baseline F1 | ~0.026 → **11x better** |

Training progression (each run fixed a different issue):

| Run | F1 | What was wrong |
|---|---|---|
| 1 | 0.060 | alpha=0.987 → predicted everything positive |
| 2 | 0.122 | no balanced sampler |
| 3 | 0.197 | LR restart spikes |
| 4 | 0.237 | 5 of 19 features were always-zero |
| 5 | 0.282 | LR still in warmup when early stopping fired |
| 6 | **0.287** | stopped at epoch 4 of 24 |

---

## Running on Colab (T4 GPU)

**One-time setup:**

1. Run locally: `python zip_for_colab.py` → creates `tp2_project_code.zip`
2. Upload to Google Drive at `MyDrive/tp2_project/`:
   - `tp2_project_code.zip` (108 KB — code)
   - `tp2_equity.db` (448 MB — pipeline data)
   - `equity_dataset.npz` (229 MB — pre-built ML features)
   - `equity_metadata.json` (49 MB — sample metadata)
3. Open `train.ipynb` in Colab → Runtime → T4 GPU
4. Run all cells top to bottom (~45–60 min)

**To update code after changes:**
```bash
python zip_for_colab.py          # re-zip
# upload new zip to Drive, then in Colab:
import shutil; shutil.rmtree('/content/tp2_project', ignore_errors=True)
# re-run Cell 1
```

---

## Next Steps

1. **Finish SPX training** — let the model train full 40 epochs without early stopping; expect F1 ~0.32–0.40
2. **Crypto transfer learning** — load `checkpoints/equity/best_model.pt`, freeze encoder layers, fine-tune on BTC/ETH/SOL data (Indu's Deribit data + Omkar's Databento pipeline)
3. **Rolling window** — Jyotir's rolling window evaluation (3-month train, 15-day predict)
4. **Cross-asset comparison** — violation rates, reversion speed, feature importance: equity vs crypto

---

## Key Files for Each Team Member

**Omkar** (pipeline, equity data):
- `collector/databento.py`, `collector/spx_spot.py`
- `ml/build_equity_dataset.py`
- `train.ipynb`

**Indu** (crypto data, Deribit):
- `collector/deribit.py`, `collector/live.py`
- `processor/cleaner.py` — equity filter fixes (zero spread, nullable OI)
- `storage/db.py`

**Jyotir** (transfer learning, rolling window):
- `ml/transformer.py` — DS-CAT architecture
- `ml/train.py` — training loop, focal loss
- `ml/dataset.py` — chronological split, balanced sampler
- `config.py` — `ROLLING_WINDOW_MONTHS`, `CRYPTO_FINETUNE_MONTHS`

---

## Common Issues

**`FileNotFoundError` in Colab:** Files must be in `MyDrive/tp2_project/` (subfolder), not Drive root.

**`IN_COLAB = False` on web Colab:** Use `try: import google.colab` not `'google.colab' in str(sys.modules)`.

**Model predicts everything positive (R=0.99, P=0.01):** Alpha too high. Fixed in `compute_focal_alpha()` — uses sqrt ratio capped at [0.6, 0.9].

**Val F1 flat at ~0.12:** Expected. Val period is Oct–Dec 2025 (higher volatility regime). Test F1 is ~2× val F1 consistently — the model generalizes to the test period (Jan–Mar 2026).
