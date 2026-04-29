from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.loader import get_path, load_config


def new_data_available() -> bool:
    """
    Check if new raw batch data is available for processing.

    Returns True if at least one new batch file exists.
    """
    batch_dir = Path(get_path("raw_data")) / "new_batches"
    if not batch_dir.exists():
        return False

    return any(batch_dir.glob("*.csv"))


def drift_detected() -> bool:
    """
    Check if feature drift has been detected in recent monitoring runs.

    Returns True if any drift flag is present in the drift history.
    """
    drift_file = Path(get_path("monitoring")) / "feature_drift_history.parquet"

    if not drift_file.exists():
        return False

    df = pd.read_parquet(drift_file)

    if df.empty:
        return False

    if "drift_detected" not in df.columns:
        return False

    latest = df.iloc[-1]
    return bool(latest["drift_detected"])


def performance_degraded() -> bool:
    """
    Check if churn classification performance dropped below configured thresholds.

    Thresholds are loaded from monitoring.yaml.
    """
    cfg = load_config("monitoring.yaml")
    thresholds = cfg.get("performance", {}).get("retrain_thresholds", {})

    min_f1 = thresholds.get("min_f1", 0.60)
    min_recall = thresholds.get("min_recall", 0.65)
    min_roc_auc = thresholds.get("min_roc_auc", 0.75)
    max_brier_score = thresholds.get("max_brier_score", 0.22)

    performance_file = Path(get_path("monitoring")) / "churn_performance_history.parquet"

    if not performance_file.exists():
        return False

    df = pd.read_parquet(performance_file)

    if df.empty:
        return False

    latest = df.sort_values("computed_at").iloc[-1]

    checks = []

    if "f1" in latest and pd.notna(latest["f1"]):
        checks.append(latest["f1"] < min_f1)

    if "recall" in latest and pd.notna(latest["recall"]):
        checks.append(latest["recall"] < min_recall)

    if "roc_auc" in latest and pd.notna(latest["roc_auc"]):
        checks.append(latest["roc_auc"] < min_roc_auc)

    if "brier_score" in latest and pd.notna(latest["brier_score"]):
        checks.append(latest["brier_score"] > max_brier_score)

    return any(checks)


def business_degraded() -> bool:
    """
    Check if expected business profit dropped below the configured threshold.

    Uses the latest churn performance monitoring snapshot.
    """
    cfg = load_config("monitoring.yaml")
    business_cfg = cfg.get("business", {})

    min_expected_profit = business_cfg.get("min_expected_profit", 0.0)

    performance_file = Path(get_path("monitoring")) / "churn_performance_history.parquet"

    if not performance_file.exists():
        return False

    df = pd.read_parquet(performance_file)

    if df.empty:
        return False

    latest = df.sort_values("computed_at").iloc[-1]

    profit_col = "business_expected_profit"

    if profit_col not in latest or pd.isna(latest[profit_col]):
        return False

    return latest[profit_col] < min_expected_profit


def should_retrain() -> bool:
    """
    Decide whether model retraining should be triggered.

    Combines new data, drift, model performance, and business signals.
    """
    return (
        new_data_available()
        or drift_detected()
        or performance_degraded()
        or business_degraded()
    )