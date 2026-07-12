#!/usr/bin/env python3
"""Run controlled TP2-MV-ALiBi transformer ablations.

This script is intentionally separate from 03_train_mv_alibi_transformer.py.
It reuses the same data construction logic, but exposes the professor-requested
training controls: checkpoint metric, learning rate, warm-up/scheduler, weight
decay, gradient clipping, model capacity, history length, dropout, and seed.
Each run writes into TP2MVTransformer/ablation_runs/<tag>/ so variants can run
in parallel without overwriting the main transformer artifacts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "TP2MVTransformer" / "03_train_mv_alibi_transformer.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("tp2_mv_alibi_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base transformer module from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


class FlexibleMVAlibiClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int,
        heads: int,
        distance_dims: int,
        input_dropout: float,
        head_dropout: float,
        use_return_head: bool,
        use_distance_bias: bool,
        use_strategy_bonus: bool,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.use_return_head = use_return_head
        self.use_distance_bias = use_distance_bias
        self.use_strategy_bonus = use_strategy_bonus

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(input_dropout),
            nn.Linear(hidden, hidden),
        )
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

        self.raw_slopes = nn.Parameter(torch.full((heads, distance_dims), -1.2))
        self.same_strategy_bonus = nn.Parameter(torch.zeros(heads))
        self.cross_strategy_bonus = nn.Parameter(torch.zeros(heads))

        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )
        if use_return_head:
            self.ret_head = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
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
        if self.use_distance_bias:
            slopes = torch.nn.functional.softplus(self.raw_slopes)
            penalty = torch.einsum("hD,blD->bhl", slopes, distances[:, :, :5])
            logits = logits - penalty

        if self.use_strategy_bonus:
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


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, bce_criterion, huber_criterion, device, lambda_return, clip_norm):
    model.train()
    losses, grad_norms = [], []
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
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm))
    return float(np.mean(losses)), float(np.mean(grad_norms))


@torch.no_grad()
def evaluate(model, loader, bce_criterion, huber_criterion, device, lambda_return):
    model.eval()
    losses, probs, rets, labels = [], [], [], []
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
        probs.append(torch.sigmoid(cls_logit).detach().cpu().numpy())
        rets.append(ret_pred.detach().cpu().numpy() if ret_pred is not None else np.zeros(len(cls_logit)))
        labels.append(batch["y"].detach().cpu().numpy())
    prob = np.concatenate(probs)
    ret = np.concatenate(rets)
    y = np.concatenate(labels)
    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y, prob))
    except ValueError:
        auprc = float("nan")
    try:
        brier = float(brier_score_loss(y, prob))
    except ValueError:
        brier = float("nan")
    return float(np.mean(losses)), auc, auprc, brier, prob, ret


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1 else prob <= right)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(prob[mask].mean() - y_true[mask].mean()))
    return ece


def apply_calibration(val_prob: np.ndarray, val_y: np.ndarray, target_prob: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return target_prob.astype(np.float32)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_prob, val_y)
        return iso.predict(target_prob).astype(np.float32)
    raise ValueError(f"Unknown calibration method: {method}")


def warmup_cosine_lambda(epoch_index: int, epochs: int, warmup_pct: float) -> float:
    warmup_epochs = max(1, int(round(epochs * warmup_pct)))
    if epoch_index < warmup_epochs:
        return float(epoch_index + 1) / float(warmup_epochs)
    progress = (epoch_index - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def metric_value(metric: str, val_loss: float, val_auc: float) -> float:
    return -val_loss if metric == "val_loss" else val_auc


def is_improvement(metric: str, candidate: float, best: float) -> bool:
    if not np.isfinite(candidate):
        return False
    return candidate > best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TP2-MV-ALiBi transformer ablation.")
    parser.add_argument("--tag", required=True, help="Output tag under TP2MVTransformer/ablation_runs/<tag>.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=64)
    parser.add_argument("--input-dropout", type=float, default=0.10)
    parser.add_argument("--head-dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-return", type=float, default=0.02)
    parser.add_argument("--checkpoint-metric", choices=["val_loss", "val_auc"], default="val_loss")
    parser.add_argument("--scheduler", choices=["none", "plateau", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--scheduler-metric", choices=["val_loss", "val_auc"], default="val_loss")
    parser.add_argument("--warmup-pct", type=float, default=0.10)
    parser.add_argument("--calibration", choices=["none", "isotonic"], default="isotonic")
    parser.add_argument("--no-distance-bias", action="store_true")
    parser.add_argument("--no-strategy-bonus", action="store_true")
    parser.add_argument("--fold-years", nargs="*", type=int, help="Optional subset of test years, e.g. --fold-years 2022 2025.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base.MAX_HISTORY = args.max_history
    out_dir = ROOT / "TP2MVTransformer" / "ablation_runs" / args.tag
    table_dir = out_dir / "tables"
    data_dir = out_dir / "data"
    table_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Using device: {device}", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    data = pd.read_parquet(base.TOKEN_PATH)
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["signal_year"] = data["signal_date"].dt.year
    data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)

    feature_dict = pd.read_csv(base.FEATURE_DICT_PATH)
    feature_cols = (
        feature_dict[feature_dict["selected_for_transformer"].astype(bool)]
        .sort_values("selected_rank")["feature"]
        .tolist()
    )
    if "signal_year" in feature_cols:
        raise RuntimeError("signal_year leaked into feature_cols")
    block_indices = {
        block: np.array([i for i, col in enumerate(feature_cols) if base.block_for_feature(col) == block], dtype=int)
        for block in ["surface", "short_d", "liquidity", "lifecycle"]
    }

    folds = [f for f in base.FOLDS if not args.fold_years or f["test"][0] in set(args.fold_years)]
    fold_rows, train_rows, prediction_rows, val_prediction_rows = [], [], [], []

    for fold in folds:
        fold_name = fold["fold"]
        test_year = fold["test"][0]
        print(f"\n{'=' * 60}\nFold {fold_name}", flush=True)

        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()

        scaler = StandardScaler()
        scaler.fit(base.safe_numeric_frame(train, feature_cols))
        train_arrays = base.make_arrays(train, train, feature_cols, scaler, block_indices)
        val_history = data[data["signal_year"].isin(fold["train"])].copy()
        val_arrays = base.make_arrays(val, val_history, feature_cols, scaler, block_indices)
        test_history = data[data["signal_year"].isin(fold["train"] + fold["val"])].copy()
        test_arrays = base.make_arrays(test, test_history, feature_cols, scaler, block_indices)
        print(
            f"  history coverage: train={train_arrays.hist_mask.mean():.3f} "
            f"val={val_arrays.hist_mask.mean():.3f} test={test_arrays.hist_mask.mean():.3f}",
            flush=True,
        )

        pos_rate = float(train["target_profitable"].mean())
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32, device=device)
        model = FlexibleMVAlibiClassifier(
            n_features=len(feature_cols),
            hidden=args.hidden,
            heads=args.heads,
            distance_dims=5,
            input_dropout=args.input_dropout,
            head_dropout=args.head_dropout,
            use_return_head=(args.lambda_return > 0),
            use_distance_bias=not args.no_distance_bias,
            use_strategy_bonus=not args.no_strategy_bonus,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler == "plateau":
            mode = "min" if args.scheduler_metric == "val_loss" else "max"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=3, min_lr=1e-6)
        elif args.scheduler == "warmup_cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: warmup_cosine_lambda(epoch, args.epochs, args.warmup_pct),
            )
        else:
            scheduler = None

        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        huber_criterion = nn.HuberLoss(delta=0.5)
        train_loader = DataLoader(base.TP2HistoryDataset(train_arrays), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(base.TP2HistoryDataset(val_arrays), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(base.TP2HistoryDataset(test_arrays), batch_size=args.batch_size, shuffle=False)

        best_score = -np.inf
        best_state = None
        best_epoch = 0
        best_val_loss = float("nan")
        best_val_auc = float("nan")
        stale = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, grad_norm = train_epoch(
                model, train_loader, optimizer, bce_criterion, huber_criterion,
                device, args.lambda_return, args.clip_norm,
            )
            val_loss, val_auc, val_auprc, val_brier_raw, val_prob_raw, _ = evaluate(
                model, val_loader, bce_criterion, huber_criterion, device, args.lambda_return
            )
            row = {
                "tag": args.tag,
                "fold": fold_name,
                "test_year": test_year,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_auprc": val_auprc,
                "val_brier_raw": val_brier_raw,
                "mean_grad_norm_pre_clip": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            train_rows.append(row)
            print(
                f"  epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auc={val_auc:.4f} grad_norm={grad_norm:.3f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

            score = metric_value(args.checkpoint_metric, val_loss, val_auc)
            if is_improvement(args.checkpoint_metric, score, best_score):
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_auc = val_auc
                stale = 0
            else:
                stale += 1

            if args.scheduler == "plateau" and scheduler is not None:
                scheduler.step(val_loss if args.scheduler_metric == "val_loss" else val_auc)
            elif args.scheduler == "warmup_cosine" and scheduler is not None:
                scheduler.step()

            if stale >= args.patience:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_loss, val_auc, val_auprc, val_brier_raw, val_prob_raw, val_ret_pred = evaluate(
            model, val_loader, bce_criterion, huber_criterion, device, args.lambda_return
        )
        test_loss, test_auc_raw, test_auprc_raw, test_brier_raw, test_prob_raw, test_ret_pred = evaluate(
            model, test_loader, bce_criterion, huber_criterion, device, args.lambda_return
        )
        val_prob = apply_calibration(val_prob_raw, val_arrays.y, val_prob_raw, args.calibration)
        test_prob = apply_calibration(val_prob_raw, val_arrays.y, test_prob_raw, args.calibration)
        val_brier = float(brier_score_loss(val_arrays.y, val_prob))
        test_brier = float(brier_score_loss(test_arrays.y, test_prob))
        val_ece = expected_calibration_error(val_arrays.y, val_prob)
        test_ece = expected_calibration_error(test_arrays.y, test_prob)

        val_tokens = val_arrays.meta.copy()
        test_tokens = test_arrays.meta.copy()
        for tokens, prob, raw_prob, ret_pred in [
            (val_tokens, val_prob, val_prob_raw, val_ret_pred),
            (test_tokens, test_prob, test_prob_raw, test_ret_pred),
        ]:
            tokens["predicted_prob"] = prob
            tokens["predicted_prob_raw"] = raw_prob
            tokens["predicted_return"] = ret_pred

        threshold = base.tune_threshold(val_tokens)
        val_selected = base.select_episode_actions(val_tokens, threshold)
        test_selected = base.select_episode_actions(test_tokens, threshold)
        val_metrics = base.summarize(val_selected, threshold)
        test_metrics = base.summarize(test_selected, threshold)

        test_tokens["fold"] = fold_name
        test_tokens["selected"] = test_tokens["episode_id"].isin(test_selected["episode_id"]) & (
            test_tokens["strategy"].eq(test_tokens["episode_id"].map(test_selected.set_index("episode_id")["strategy"].to_dict()))
        )
        val_tokens["fold"] = fold_name
        prediction_rows.append(test_tokens)
        val_prediction_rows.append(val_tokens)

        fold_rows.append({
            "tag": args.tag,
            "fold": fold_name,
            "train_years": ",".join(map(str, fold["train"])),
            "val_year": fold["val"][0],
            "test_year": test_year,
            "feature_count": len(feature_cols),
            "max_history_tokens": args.max_history,
            "hidden": args.hidden,
            "heads": args.heads,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "input_dropout": args.input_dropout,
            "head_dropout": args.head_dropout,
            "checkpoint_metric": args.checkpoint_metric,
            "scheduler": args.scheduler,
            "scheduler_metric": args.scheduler_metric,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_auc": best_val_auc,
            "val_loss_final": val_loss,
            "test_loss": test_loss,
            "val_auc": val_auc,
            "test_auc_raw": test_auc_raw,
            "test_auprc_raw": test_auprc_raw,
            "val_brier": val_brier,
            "test_brier": test_brier,
            "val_ece": val_ece,
            "test_ece": test_ece,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
        print(
            f"  threshold={threshold:.3f} test_pnl=${test_metrics['norm_total_pnl']:,.0f} "
            f"test_n={test_metrics['n_selected']:,} sharpe={test_metrics['daily_sharpe']:.2f}",
            flush=True,
        )

    results = pd.DataFrame(fold_rows)
    training = pd.DataFrame(train_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    val_predictions = pd.concat(val_prediction_rows, ignore_index=True)
    selected = predictions[predictions["selected"]].copy()
    summary = {
        "tag": args.tag,
        "args": vars(args),
        "device": str(device),
        "total_selected": int(len(selected)),
        "total_norm_pnl": float(selected["target_norm_pnl"].sum()) if len(selected) else 0.0,
        "positive_years": int((results["test_norm_total_pnl"] > 0).sum()),
        "total_years": int(len(results)),
        "combined_daily_sharpe": base.daily_sharpe(selected),
        "combined_daily_sortino": base.daily_sortino(selected),
        "combined_max_drawdown": base.daily_drawdown(selected),
        "combined_win_rate": float((selected["target_norm_pnl"] > 0).mean()) if len(selected) else float("nan"),
    }

    results.to_csv(table_dir / "ablation_yearly_results.csv", index=False)
    training.to_csv(table_dir / "ablation_training_history.csv", index=False)
    predictions.to_parquet(data_dir / "ablation_test_predictions.parquet", index=False)
    val_predictions.to_parquet(data_dir / "ablation_val_predictions.parquet", index=False)
    selected.to_parquet(data_dir / "ablation_selected_trades.parquet", index=False)
    with (table_dir / "ablation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nAblation summary:")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
