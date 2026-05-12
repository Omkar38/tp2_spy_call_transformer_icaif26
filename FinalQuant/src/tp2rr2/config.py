from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    archive: Path = PROJECT_ROOT / "archive"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    features: Path = PROJECT_ROOT / "data" / "features"
    labels: Path = PROJECT_ROOT / "data" / "labels"
    models: Path = PROJECT_ROOT / "data" / "models"
    reports: Path = PROJECT_ROOT / "reports"
    spy_close: Path = PROJECT_ROOT / "data" / "raw" / "spy_daily.csv"


@dataclass(frozen=True)
class CleanConfig:
    min_dte: int = 1
    max_dte: int = 60
    min_abs_delta: float = 0.01
    max_abs_delta: float = 0.99
    max_spread_pct: float = 0.50
    require_positive_size_if_present: bool = False


@dataclass(frozen=True)
class DetectionConfig:
    min_normalized_severity: float = 0.0001
    max_expirations_per_day: int | None = None
    max_strikes_per_expiration: int = 80
    strike_selection: str = "near_underlying"
    n_jobs: int = 1


@dataclass(frozen=True)
class LabelConfig:
    horizon_trading_days: int = 5
    secondary_horizon_trading_days: int = 1
    commission_per_contract: float = 0.65
    option_multiplier: float = 100.0
    max_label_spread_pct: float = 0.50


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    validation_start: str = "2023-01-01"
    validation_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"


@dataclass(frozen=True)
class BacktestConfig:
    prediction_threshold: float = 0.0
    min_normalized_severity: float = 0.0005
    max_spread_pct: float = 0.25
    max_avg_spread_pct: float = 0.15
    min_volume: int = 1
    min_open_interest: int = 1
    max_trades_per_day: int = 25
    max_contract_overlap_per_day: int = 3
    target_gross_exposure: float = 1000.0


@dataclass(frozen=True)
class PipelineConfig:
    paths: Paths = Paths()
    clean: CleanConfig = CleanConfig()
    detection: DetectionConfig = DetectionConfig()
    labels: LabelConfig = LabelConfig()
    splits: SplitConfig = SplitConfig()
    backtest: BacktestConfig = BacktestConfig()


CONFIG = PipelineConfig()
