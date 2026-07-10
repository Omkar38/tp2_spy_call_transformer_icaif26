"""
Train the TP2-MV-ALiBi transformer classifier (v2 — increased capacity).

Changes from v1:
- hidden 32 → 64, MAX_HISTORY 32 → 64, EPOCHS 6 → 25, PATIENCE 3 → 5
- Optional Huber return auxiliary head (lambda=0.1, diagnostics only).
  Trading decisions still use binary probability only.
- LR scheduler: ReduceLROnPlateau on val AUC.
- Isotonic calibration fitted on val probs before threshold tuning.
- Year-wise diagnostics table + 2022 fold detailed breakdown saved separately.

Design rules (unchanged):
- strategies restricted to T1/K2
- no signal_year feature
- no booster predictions, leaves, scores, or SHAP values as model inputs
- thresholds tuned on validation year only; test year evaluated once
- pairwise ranking loss intentionally excluded (cross-episode ranking not
  conceptually clean for this trading problem)
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATH = DATA_DIR / "tp2_t1k2_token_dataset.parquet"
FEATURE_DICT_PATH = TABLE_DIR / "feature_dictionary.csv"

NORM_EXPOSURE = 1_000.0
MAX_HISTORY = 64
BATCH_SIZE = 512
EPOCHS = 25
PATIENCE = 5
LR = 5e-4
WEIGHT_DECAY = 1e-4
LAMBDA_RETURN = 0.02         # auxiliary Huber weight; 0.0 disables return head
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.025)
MIN_TRADES_FOR_THRESHOLD = 30

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "train": [2017, 2018, 2019], "val": [2020], "test": [2021]},
    {"fold": "2018_2020__val2021__test2022", "train": [2018, 2019, 2020], "val": [2021], "test": [2022]},
    {"fold": "2019_2021__val2022__test2023", "train": [2019, 2020, 2021], "val": [2022], "test": [2023]},
    {"fold": "2020_2022__val2023__test2024", "train": [2020, 2021, 2022], "val": [2023], "test": [2024]},
    {"fold": "2021_2023__val2024__test2025", "train": [2021, 2022, 2023], "val": [2024], "test": [2025]},
    {"fold": "2022_2024__val2025__test2026", "train": [2022, 2023, 2024], "val": [2025], "test": [2026]},
]


# ── helpers ────────────────────────────────────────────────────────────────────

def block_for_feature(name: str) -> str:
    lower = name.lower()
    if "moneyness" in lower or "dte" in lower or "strike" in lower or "term_gap" in lower or "distance" in lower or lower in {"feature_f_t1", "feature_f_t2"}:
        return "surface"
    if lower.startswith("feature_d_") or "_d_" in lower:
        return "short_d"
    if "spread" in lower or "volume" in lower or "open_interest" in lower or "zero_" in lower or "bidask" in lower:
        return "liquidity"
    if lower.startswith("lc_") or "violation_age" in lower or "seen_previous" in lower:
        return "lifecycle"
    return "other"


def safe_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class FoldArrays:
    x: np.ndarray
    y: np.ndarray
    ret: np.ndarray          # normalized return, for auxiliary head
    pnl: np.ndarray
    dates: np.ndarray
    meta: pd.DataFrame
    hist_x: np.ndarray
    hist_mask: np.ndarray
    distances: np.ndarray


class TP2HistoryDataset(Dataset):
    def __init__(self, arrays: FoldArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.arrays.x[idx], dtype=torch.float32),
            "hist_x": torch.tensor(self.arrays.hist_x[idx], dtype=torch.float32),
            "hist_mask": torch.tensor(self.arrays.hist_mask[idx], dtype=torch.bool),
            "distances": torch.tensor(self.arrays.distances[idx], dtype=torch.float32),
            "y": torch.tensor(self.arrays.y[idx], dtype=torch.float32),
            "ret": torch.tensor(self.arrays.ret[idx], dtype=torch.float32),
        }


# ── model ──────────────────────────────────────────────────────────────────────

class MVAlibiClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, heads: int = 4, distance_dims: int = 5, use_return_head: bool = True):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.use_return_head = use_return_head

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
        )
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

        # learned slopes; softplus ensures they stay positive
        self.raw_slopes = nn.Parameter(torch.full((heads, distance_dims), -1.2))
        self.same_strategy_bonus = nn.Parameter(torch.zeros(heads))
        self.cross_strategy_bonus = nn.Parameter(torch.zeros(heads))

        # binary classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

        # auxiliary return head (diagnostics only; not used for trading decisions)
        if use_return_head:
            self.ret_head = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        hist_x: torch.Tensor,
        hist_mask: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, hist_len, _ = hist_x.shape
        cur = self.input_proj(x)
        hist = self.input_proj(hist_x.reshape(batch * hist_len, -1)).reshape(batch, hist_len, self.hidden)

        q = self.q(cur).reshape(batch, self.heads, self.head_dim)
        k = self.k(hist).reshape(batch, hist_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(hist).reshape(batch, hist_len, self.heads, self.head_dim).transpose(1, 2)

        logits = torch.einsum("bhd,bhld->bhl", q, k) / math.sqrt(self.head_dim)
        slopes = torch.nn.functional.softplus(self.raw_slopes)
        penalty = torch.einsum("hD,blD->bhl", slopes, distances[:, :, :5])
        logits = logits - penalty

        same_strategy = distances[:, :, 5].unsqueeze(1)
        logits = logits + same_strategy * self.same_strategy_bonus.view(1, self.heads, 1)
        logits = logits + (1.0 - same_strategy) * self.cross_strategy_bonus.view(1, self.heads, 1)

        logits = logits.masked_fill(~hist_mask.unsqueeze(1), -1.0e9)
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bhl,bhld->bhd", weights, v).reshape(batch, self.hidden)

        combined = torch.cat([cur, context], dim=1)
        cls_logit = self.cls_head(combined).squeeze(1)
        ret_pred = self.ret_head(combined).squeeze(1) if self.use_return_head else None
        return cls_logit, ret_pred


# ── history construction ───────────────────────────────────────────────────────

def build_history_arrays(
    current: pd.DataFrame,
    history: pd.DataFrame,
    x_current: np.ndarray,
    x_history: np.ndarray,
    feature_cols: list[str],
    block_indices: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist_dates = pd.to_datetime(history["signal_date"]).astype("int64").to_numpy() // 86_400_000_000_000
    cur_dates = pd.to_datetime(current["signal_date"]).astype("int64").to_numpy() // 86_400_000_000_000
    hist_strategy = history["strategy_id"].to_numpy(dtype=np.float32)
    cur_strategy = current["strategy_id"].to_numpy(dtype=np.float32)

    hist_x = np.zeros((len(current), MAX_HISTORY, len(feature_cols)), dtype=np.float32)
    hist_mask = np.zeros((len(current), MAX_HISTORY), dtype=bool)
    distances = np.zeros((len(current), MAX_HISTORY, 6), dtype=np.float32)

    for row_idx, day in enumerate(cur_dates):
        end = np.searchsorted(hist_dates, day, side="left")
        if end <= 0:
            continue
        start = max(0, end - MAX_HISTORY)
        idx = np.arange(start, end)
        target_slice = slice(MAX_HISTORY - len(idx), MAX_HISTORY)
        hx = x_history[idx]
        hist_x[row_idx, target_slice, :] = hx
        hist_mask[row_idx, target_slice] = True

        distances[row_idx, target_slice, 0] = np.log1p(np.maximum(day - hist_dates[idx], 0))
        for dist_pos, block_name in enumerate(["surface", "short_d", "liquidity", "lifecycle"], start=1):
            cols = block_indices.get(block_name, np.array([], dtype=int))
            if len(cols) == 0:
                continue
            distances[row_idx, target_slice, dist_pos] = np.abs(x_current[row_idx, cols] - hx[:, cols]).mean(axis=1)
        distances[row_idx, target_slice, 5] = (cur_strategy[row_idx] == hist_strategy[idx]).astype(np.float32)

    return hist_x, hist_mask, distances


def make_arrays(
    rows: pd.DataFrame,
    history_rows: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    block_indices: dict[str, np.ndarray],
) -> FoldArrays:
    x = scaler.transform(safe_numeric_frame(rows, feature_cols)).astype(np.float32)
    hist_x_base = scaler.transform(safe_numeric_frame(history_rows, feature_cols)).astype(np.float32)
    hist_x, hist_mask, distances = build_history_arrays(rows, history_rows, x, hist_x_base, feature_cols, block_indices)
    return FoldArrays(
        x=x,
        y=rows["target_profitable"].to_numpy(dtype=np.float32),
        ret=rows["target_return"].to_numpy(dtype=np.float32),
        pnl=rows["target_norm_pnl"].to_numpy(dtype=np.float32),
        dates=pd.to_datetime(rows["signal_date"]).to_numpy(),
        meta=rows[["episode_id", "signal_date", "signal_year", "strategy", "target_profitable", "target_return", "target_norm_pnl"]].copy(),
        hist_x=hist_x,
        hist_mask=hist_mask,
        distances=distances,
    )


DISTANCE_NAMES = ["time", "surface", "short_d", "liquidity", "lifecycle"]


def distance_summary(arrays: FoldArrays, fold_name: str, split: str) -> list[dict]:
    rows = []
    if arrays.distances.size == 0:
        return rows
    valid = arrays.hist_mask
    if not valid.any():
        for idx, name in enumerate(DISTANCE_NAMES):
            rows.append({
                "fold": fold_name,
                "split": split,
                "distance": name,
                "n": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "p01": float("nan"),
                "p50": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            })
        return rows
    for idx, name in enumerate(DISTANCE_NAMES):
        vals = arrays.distances[:, :, idx][valid]
        rows.append({
            "fold": fold_name,
            "split": split,
            "distance": name,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p01": float(np.quantile(vals, 0.01)),
            "p50": float(np.quantile(vals, 0.50)),
            "p99": float(np.quantile(vals, 0.99)),
            "max": float(np.max(vals)),
        })
    return rows


def attention_parameter_rows(model: MVAlibiClassifier, fold_name: str, test_year: int) -> list[dict]:
    slopes = torch.nn.functional.softplus(model.raw_slopes).detach().cpu().numpy()
    same = model.same_strategy_bonus.detach().cpu().numpy()
    cross = model.cross_strategy_bonus.detach().cpu().numpy()
    rows = []
    for head in range(slopes.shape[0]):
        row = {
            "fold": fold_name,
            "test_year": test_year,
            "head": int(head),
            "same_strategy_bonus": float(same[head]),
            "cross_strategy_bonus": float(cross[head]),
        }
        for idx, name in enumerate(DISTANCE_NAMES):
            row[f"slope_{name}"] = float(slopes[head, idx])
        rows.append(row)
    return rows


# ── training ───────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_criterion: nn.Module,
    huber_criterion: nn.Module,
    device: torch.device,
    lambda_return: float,
) -> tuple[float, float]:
    model.train()
    losses = []
    grad_norms = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        cls_logit, ret_pred = model(
            batch["x"].to(device),
            batch["hist_x"].to(device),
            batch["hist_mask"].to(device),
            batch["distances"].to(device),
        )
        loss = bce_criterion(cls_logit, batch["y"].to(device))
        if lambda_return > 0 and ret_pred is not None:
            loss = loss + lambda_return * huber_criterion(ret_pred, batch["ret"].to(device))
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm))
    return float(np.mean(losses)), float(np.mean(grad_norms))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    bce_criterion: nn.Module,
    huber_criterion: nn.Module,
    device: torch.device,
    lambda_return: float,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        cls_logit, ret_pred = model(
            batch["x"].to(device),
            batch["hist_x"].to(device),
            batch["hist_mask"].to(device),
            batch["distances"].to(device),
        )
        loss = bce_criterion(cls_logit, batch["y"].to(device))
        if lambda_return > 0 and ret_pred is not None:
            loss = loss + lambda_return * huber_criterion(ret_pred, batch["ret"].to(device))
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, rets = [], []
    for batch in loader:
        cls_logit, ret_pred = model(
            batch["x"].to(device),
            batch["hist_x"].to(device),
            batch["hist_mask"].to(device),
            batch["distances"].to(device),
        )
        probs.append(torch.sigmoid(cls_logit).detach().cpu().numpy())
        if ret_pred is not None:
            rets.append(ret_pred.detach().cpu().numpy())
        else:
            rets.append(np.zeros(len(cls_logit)))
    return np.concatenate(probs), np.concatenate(rets)


# ── calibration ────────────────────────────────────────────────────────────────

def fit_isotonic_calibration(val_probs: np.ndarray, val_labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, val_labels)
    return iso


def apply_calibration(iso: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    return iso.predict(probs).astype(np.float32)


# ── selection and evaluation ───────────────────────────────────────────────────

def select_episode_actions(token_rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = token_rows[token_rows["predicted_prob"] >= threshold].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
        .groupby("episode_id", as_index=False)
        .first()
    )


def tune_threshold(validation_tokens: pd.DataFrame) -> float:
    best_threshold = 0.5
    best_pnl = -np.inf
    for threshold in THRESHOLD_GRID:
        selected = select_episode_actions(validation_tokens, float(threshold))
        if len(selected) < MIN_TRADES_FOR_THRESHOLD:
            continue
        pnl = float(selected["target_norm_pnl"].sum())
        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = float(threshold)
    return best_threshold


def daily_sharpe(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def daily_sortino(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    if len(daily) < 2:
        return float("nan")
    downside = daily[daily < 0.0]
    if downside.empty:
        return float("inf") if daily.mean() > 0 else float("nan")
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")
    return float(daily.mean() / downside_std * np.sqrt(252))


def daily_drawdown(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if len(drawdown) else float("nan")


def summarize(selected: pd.DataFrame, threshold: float) -> dict:
    if selected.empty:
        return {
            "threshold": threshold,
            "n_selected": 0,
            "norm_total_pnl": 0.0,
            "win_rate": float("nan"),
            "daily_sharpe": float("nan"),
            "daily_sortino": float("nan"),
            "max_drawdown": float("nan"),
            "t1_selected": 0,
            "k2_selected": 0,
        }
    dist = selected["strategy"].value_counts().to_dict()
    return {
        "threshold": threshold,
        "n_selected": int(len(selected)),
        "norm_total_pnl": float(selected["target_norm_pnl"].sum()),
        "win_rate": float((selected["target_norm_pnl"] > 0).mean()),
        "daily_sharpe": daily_sharpe(selected),
        "daily_sortino": daily_sortino(selected),
        "max_drawdown": daily_drawdown(selected),
        "t1_selected": int(dist.get("T1", 0)),
        "k2_selected": int(dist.get("K2", 0)),
    }


# ── 2022 fold diagnostics ──────────────────────────────────────────────────────

def save_fold_diagnostics(
    test_tokens: pd.DataFrame,
    selected: pd.DataFrame,
    fold_name: str,
    test_year: int,
) -> None:
    """Save detailed episode-level diagnostics for any fold. Always runs for 2022."""
    out_path = TABLE_DIR / f"fold_{test_year}_diagnostics.csv"

    tokens = test_tokens.copy()
    tokens["was_selected"] = tokens["episode_id"].isin(selected["episode_id"]) & (
        tokens["strategy"].eq(
            tokens["episode_id"].map(
                selected.set_index("episode_id")["strategy"].to_dict()
            )
        )
    )
    sel = tokens[tokens["was_selected"]].copy()
    if sel.empty:
        return

    # loss vs win breakdown
    sel["outcome"] = np.where(sel["target_norm_pnl"] > 0, "win", "loss")

    diag = sel.groupby(["strategy", "outcome"]).agg(
        n=("target_norm_pnl", "count"),
        total_pnl=("target_norm_pnl", "sum"),
        mean_pnl=("target_norm_pnl", "mean"),
        median_pnl=("target_norm_pnl", "median"),
        p10_pnl=("target_norm_pnl", lambda x: x.quantile(0.10)),
        p90_pnl=("target_norm_pnl", lambda x: x.quantile(0.90)),
        mean_prob=("predicted_prob", "mean"),
    ).reset_index()
    diag.to_csv(out_path, index=False)

    # top-20 worst losses
    worst_path = TABLE_DIR / f"fold_{test_year}_worst_losses.csv"
    worst = sel.nsmallest(20, "target_norm_pnl")[
        ["episode_id", "signal_date", "strategy", "predicted_prob", "predicted_return", "target_norm_pnl", "target_return"]
    ]
    worst.to_csv(worst_path, index=False)

    print(f"  [{test_year}] strategy/outcome breakdown:")
    print(diag.to_string(index=False))
    top_loss = float(sel["target_norm_pnl"].min())
    print(f"  [{test_year}] single worst loss: ${top_loss:,.0f}")
    loss_pnl = float(sel[sel["target_norm_pnl"] <= 0]["target_norm_pnl"].sum())
    win_pnl = float(sel[sel["target_norm_pnl"] > 0]["target_norm_pnl"].sum())
    print(f"  [{test_year}] total win P&L: ${win_pnl:,.0f} | total loss P&L: ${loss_pnl:,.0f}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = choose_device()
    print(f"Using device: {device}", flush=True)
    print(f"MAX_HISTORY={MAX_HISTORY} | EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | hidden=64 | LAMBDA_RETURN={LAMBDA_RETURN}", flush=True)

    data = pd.read_parquet(TOKEN_PATH)
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["signal_year"] = data["signal_date"].dt.year
    data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)

    feature_dict = pd.read_csv(FEATURE_DICT_PATH)
    feature_cols = (
        feature_dict[feature_dict["selected_for_transformer"].astype(bool)]
        .sort_values("selected_rank")["feature"]
        .tolist()
    )
    if "signal_year" in feature_cols:
        raise RuntimeError("signal_year leaked into feature_cols")

    block_indices = {
        block: np.array([i for i, col in enumerate(feature_cols) if block_for_feature(col) == block], dtype=int)
        for block in ["surface", "short_d", "liquidity", "lifecycle"]
    }

    fold_rows = []
    prediction_rows = []
    val_prediction_rows = []
    training_history_rows = []
    distance_rows = []
    attention_rows = []

    for fold in FOLDS:
        fold_name = fold["fold"]
        test_year = fold["test"][0]
        print(f"\n{'='*60}", flush=True)
        print(f"Fold {fold_name}", flush=True)

        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()

        scaler = StandardScaler()
        scaler.fit(safe_numeric_frame(train, feature_cols))

        train_arrays = make_arrays(train, train, feature_cols, scaler, block_indices)
        val_history = data[data["signal_year"].isin(fold["train"])].copy()
        val_arrays = make_arrays(val, val_history, feature_cols, scaler, block_indices)
        test_history = data[data["signal_year"].isin(fold["train"] + fold["val"])].copy()
        test_arrays = make_arrays(test, test_history, feature_cols, scaler, block_indices)
        distance_rows.extend(distance_summary(train_arrays, fold_name, "train"))
        distance_rows.extend(distance_summary(val_arrays, fold_name, "validation"))
        distance_rows.extend(distance_summary(test_arrays, fold_name, "test"))

        pos_rate = float(train["target_profitable"].mean())
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32, device=device)

        model = MVAlibiClassifier(
            n_features=len(feature_cols),
            hidden=64,
            heads=4,
            use_return_head=(LAMBDA_RETURN > 0),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
        )
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        huber_criterion = nn.HuberLoss(delta=0.5)

        train_loader = DataLoader(TP2HistoryDataset(train_arrays), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TP2HistoryDataset(val_arrays), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TP2HistoryDataset(test_arrays), batch_size=BATCH_SIZE, shuffle=False)

        best_auc = -np.inf
        best_state = None
        best_epoch = 0
        best_val_loss = float("nan")
        stale = 0
        for epoch in range(1, EPOCHS + 1):
            train_loss, grad_norm = train_epoch(model, train_loader, optimizer, bce_criterion, huber_criterion, device, LAMBDA_RETURN)
            val_loss = evaluate_loss(model, val_loader, bce_criterion, huber_criterion, device, LAMBDA_RETURN)
            val_prob, _ = predict(model, val_loader, device)
            try:
                val_auc = float(roc_auc_score(val_arrays.y, val_prob))
            except ValueError:
                val_auc = float("nan")
            training_history_rows.append({
                "fold": fold_name,
                "test_year": test_year,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "mean_grad_norm_pre_clip": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            print(
                f"  epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
                f"grad_norm={grad_norm:.3f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            if np.isfinite(val_auc):
                scheduler.step(val_auc)
            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        attention_rows.extend(attention_parameter_rows(model, fold_name, test_year))

        # raw predictions
        val_prob_raw, val_ret_pred = predict(model, val_loader, device)
        test_prob_raw, test_ret_pred = predict(model, test_loader, device)

        # isotonic calibration fitted on val, applied to both
        iso = fit_isotonic_calibration(val_prob_raw, val_arrays.y)
        val_prob_cal = apply_calibration(iso, val_prob_raw)
        test_prob_cal = apply_calibration(iso, test_prob_raw)

        val_tokens = val_arrays.meta.copy()
        test_tokens = test_arrays.meta.copy()
        val_tokens["predicted_prob"] = val_prob_cal
        val_tokens["predicted_prob_raw"] = val_prob_raw
        val_tokens["predicted_return"] = val_ret_pred
        test_tokens["predicted_prob"] = test_prob_cal
        test_tokens["predicted_prob_raw"] = test_prob_raw
        test_tokens["predicted_return"] = test_ret_pred

        threshold = tune_threshold(val_tokens)
        val_selected = select_episode_actions(val_tokens, threshold)
        test_selected = select_episode_actions(test_tokens, threshold)

        val_metrics = summarize(val_selected, threshold)
        test_metrics = summarize(test_selected, threshold)
        try:
            test_auc = float(roc_auc_score(test_arrays.y, test_prob_cal))
        except ValueError:
            test_auc = float("nan")

        row = {
            "fold": fold_name,
            "train_years": ",".join(map(str, fold["train"])),
            "val_year": fold["val"][0],
            "test_year": test_year,
            "feature_count": len(feature_cols),
            "max_history_tokens": MAX_HISTORY,
            "val_auc": best_auc,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_auc": test_auc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        fold_rows.append(row)

        test_tokens["fold"] = fold_name
        test_tokens["selected"] = test_tokens["episode_id"].isin(test_selected["episode_id"]) & (
            test_tokens["strategy"].eq(
                test_tokens["episode_id"].map(
                    test_selected.set_index("episode_id")["strategy"].to_dict()
                )
            )
        )
        prediction_rows.append(test_tokens)

        val_tokens["fold"] = fold_name
        val_prediction_rows.append(val_tokens)

        # diagnostics for 2022 and any negative year
        if test_year == 2022 or test_metrics["norm_total_pnl"] < 0:
            save_fold_diagnostics(test_tokens, test_selected, fold_name, test_year)

        print(
            f"  threshold={threshold:.3f} "
            f"test_pnl=${test_metrics['norm_total_pnl']:,.0f} "
            f"test_n={test_metrics['n_selected']:,} "
            f"sharpe={test_metrics['daily_sharpe']:.2f} "
            f"win_rate={test_metrics['win_rate']:.3f}",
            flush=True,
        )

    results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected = predictions[predictions["selected"]].copy()
    training_history = pd.DataFrame(training_history_rows)
    distance_diagnostics = pd.DataFrame(distance_rows)
    attention_diagnostics = pd.DataFrame(attention_rows)

    selected_daily = (
        selected.groupby("signal_date", as_index=False)["target_norm_pnl"].sum()
        if len(selected)
        else pd.DataFrame(columns=["signal_date", "target_norm_pnl"])
    )
    if len(selected_daily):
        selected_daily["signal_year"] = pd.to_datetime(selected_daily["signal_date"]).dt.year

    # year-wise diagnostics table
    yearly = results[[
        "test_year",
        "test_threshold",
        "test_n_selected",
        "test_norm_total_pnl",
        "test_win_rate",
        "test_daily_sharpe",
        "test_daily_sortino",
        "test_max_drawdown",
        "test_t1_selected",
        "test_k2_selected",
        "val_auc",
        "test_auc",
    ]].copy()
    yearly.to_csv(TABLE_DIR / "mv_alibi_v2_yearly_diagnostics.csv", index=False)
    training_history.to_csv(TABLE_DIR / "mv_alibi_v2_training_history.csv", index=False)
    distance_diagnostics.to_csv(TABLE_DIR / "mv_alibi_v2_distance_diagnostics.csv", index=False)
    attention_diagnostics.to_csv(TABLE_DIR / "mv_alibi_v2_attention_parameters.csv", index=False)
    selected.to_parquet(DATA_DIR / "mv_alibi_transformer_v2_selected_trades.parquet", index=False)
    selected.to_csv(TABLE_DIR / "mv_alibi_transformer_v2_selected_trades.csv", index=False)
    selected_daily.to_csv(TABLE_DIR / "mv_alibi_transformer_v2_selected_daily_pnl.csv", index=False)

    summary = {
        "method": "TP2-MV-ALiBi cross-attention transformer v2",
        "feature_count": len(feature_cols),
        "max_history_tokens": MAX_HISTORY,
        "hidden_dim": 64,
        "epochs_max": EPOCHS,
        "lambda_return": LAMBDA_RETURN,
        "calibration": "isotonic on val probs",
        "diagnostics": {
            "training_history_csv": str(TABLE_DIR / "mv_alibi_v2_training_history.csv"),
            "distance_diagnostics_csv": str(TABLE_DIR / "mv_alibi_v2_distance_diagnostics.csv"),
            "attention_parameters_csv": str(TABLE_DIR / "mv_alibi_v2_attention_parameters.csv"),
            "selected_daily_pnl_csv": str(TABLE_DIR / "mv_alibi_transformer_v2_selected_daily_pnl.csv"),
        },
        "total_selected": int(len(selected)),
        "total_norm_pnl": float(selected["target_norm_pnl"].sum()),
        "positive_years": int((results["test_norm_total_pnl"] > 0).sum()),
        "total_years": int(len(results)),
        "combined_daily_sharpe": daily_sharpe(selected),
        "combined_daily_sortino": daily_sortino(selected),
        "combined_max_drawdown": daily_drawdown(selected),
        "combined_win_rate": float((selected["target_norm_pnl"] > 0).mean()) if len(selected) else float("nan"),
        "device": str(device),
        "notes": [
            "Pairwise ranking loss excluded: cross-episode ranking not clean for this trading problem.",
            "Return head is auxiliary and diagnostic only; trading decisions use binary probability.",
            "Thresholds tuned on calibrated val probs; test evaluated once.",
            "Sortino is annualized from selected daily normalized P&L with zero risk-free rate.",
            "Training history includes train loss, validation loss, validation AUC, learning rate, and pre-clip gradient norm.",
            "Distance diagnostics report scale statistics for time, surface, short-D-leg, liquidity, and lifecycle distances.",
            "Attention diagnostics report learned softplus slopes and same/cross strategy bonuses by attention head.",
            "2022 fold diagnostics saved to fold_2022_diagnostics.csv and fold_2022_worst_losses.csv.",
        ],
    }

    val_predictions = pd.concat(val_prediction_rows, ignore_index=True)

    results.to_csv(TABLE_DIR / "mv_alibi_transformer_v2_results.csv", index=False)
    predictions.to_parquet(DATA_DIR / "mv_alibi_transformer_v2_test_predictions.parquet", index=False)
    val_predictions.to_parquet(DATA_DIR / "mv_alibi_transformer_v2_val_predictions.parquet", index=False)
    with open(TABLE_DIR / "mv_alibi_transformer_v2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "="*60)
    print("TP2-MV-ALiBi v2 summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nYearly diagnostics:")
    print(yearly.to_string(index=False))
    print(f"\nWrote {TABLE_DIR / 'mv_alibi_transformer_v2_results.csv'}")
    print(f"Wrote {TABLE_DIR / 'mv_alibi_v2_training_history.csv'}")
    print(f"Wrote {TABLE_DIR / 'mv_alibi_v2_distance_diagnostics.csv'}")
    print(f"Wrote {TABLE_DIR / 'mv_alibi_v2_attention_parameters.csv'}")
    print(f"Wrote {DATA_DIR / 'mv_alibi_transformer_v2_test_predictions.parquet'}")
    print(f"Wrote {DATA_DIR / 'mv_alibi_transformer_v2_val_predictions.parquet'}")
    print(f"Wrote {DATA_DIR / 'mv_alibi_transformer_v2_selected_trades.parquet'}")
    print(f"Wrote {TABLE_DIR / 'mv_alibi_transformer_v2_selected_daily_pnl.csv'}")


if __name__ == "__main__":
    main()
