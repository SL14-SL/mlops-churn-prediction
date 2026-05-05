from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.monitoring.config import get_business_settings


# -------------------------------------------------
# I/O Helpers
# -------------------------------------------------

def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported file format: {path.suffix}")


# -------------------------------------------------
# Churn Classification Metrics
# -------------------------------------------------

def _normalize_binary_target(y: pd.Series) -> pd.Series:
    """
    Supports:
    - 0/1
    - True/False
    - Yes/No
    - Churn/No Churn
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "churn": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "no churn": 0,
    }

    normalized = y.astype(str).str.strip().str.lower().map(mapping)

    if normalized.isna().any():
        bad_values = sorted(y[normalized.isna()].astype(str).unique())
        raise ValueError(f"Unknown target values: {bad_values}")

    return normalized.astype(int)


def compute_classification_metrics(
    df: pd.DataFrame,
    *,
    y_true_col: str = "Churn",
    y_proba_col: str = "churn_probability",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Computes production monitoring metrics for churn classification.

    Expected columns:
    - Churn: true label
    - churn_probability: predicted probability of churn
    """

    required = [y_true_col, y_proba_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean_df = df.dropna(subset=required).copy()

    if clean_df.empty:
        raise ValueError("No rows available after dropping missing values.")

    y_true = _normalize_binary_target(clean_df[y_true_col])
    y_proba = clean_df[y_proba_col].astype(float).clip(0.0, 1.0)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "n_samples": int(len(clean_df)),
        "threshold": float(threshold),

        # Core classification metrics
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),

        # Probability quality
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_proba),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_proba),
        "log_loss": _safe_metric(log_loss, y_true, y_proba),
        "brier_score": _safe_metric(brier_score_loss, y_true, y_proba),

        # Confusion matrix
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        # Operational monitoring
        "predicted_churn_rate": float(y_pred.mean()),
        "actual_churn_rate": float(y_true.mean()),
        "avg_churn_probability": float(y_proba.mean()),
    }

    return metrics


def _safe_metric(metric_fn, y_true: pd.Series, y_score: pd.Series) -> float | None:
    """
    Some metrics fail when only one class is present in a batch.
    In production monitoring this can happen, especially with small windows.
    """
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


# -------------------------------------------------
# Business Metrics
# -------------------------------------------------

def compute_business_metrics(
    df: pd.DataFrame,
    *,
    y_true_col: str = "Churn",
    y_proba_col: str = "churn_probability",
    action_col: str = "action",
    customer_value: float = 100.0,
    cost_contact: float = 2.0,
    cost_discount: float = 10.0,
    contact_uplift: float = 0.1,
    discount_uplift: float = 0.3,
) -> dict[str, Any]:
    """
    Estimates business impact of churn interventions.

    action values expected:
    - no_action
    - send_email
    - offer_discount
    """

    required = [y_true_col, y_proba_col, action_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean_df = df.dropna(subset=required).copy()

    if clean_df.empty:
        raise ValueError("No rows available after dropping missing values.")

    y_true = _normalize_binary_target(clean_df[y_true_col])
    y_proba = clean_df[y_proba_col].astype(float).clip(0.0, 1.0)
    actions = clean_df[action_col].astype(str)

    costs = np.select(
        [
            actions.eq("send_email"),
            actions.eq("offer_discount"),
        ],
        [
            cost_contact,
            cost_discount,
        ],
        default=0.0,
    )

    uplifts = np.select(
        [
            actions.eq("send_email"),
            actions.eq("offer_discount"),
        ],
        [
            contact_uplift,
            discount_uplift,
        ],
        default=0.0,
    )

    expected_saved_value = y_proba * customer_value * uplifts
    expected_profit = expected_saved_value - costs

    actual_intervened_churners = ((actions != "no_action") & (y_true == 1)).sum()

    return {
        "n_samples": int(len(clean_df)),
        "intervention_rate": float((actions != "no_action").mean()),
        "send_email_rate": float(actions.eq("send_email").mean()),
        "offer_discount_rate": float(actions.eq("offer_discount").mean()),
        "total_intervention_cost": float(costs.sum()),
        "expected_saved_value": float(expected_saved_value.sum()),
        "expected_profit": float(expected_profit.sum()),
        "avg_expected_profit_per_customer": float(expected_profit.mean()),
        "actual_intervened_churners": int(actual_intervened_churners),
    }


# -------------------------------------------------
# Monitoring Table Helpers
# -------------------------------------------------

def append_metrics_history(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    row = pd.DataFrame([{**metrics, "computed_at": pd.Timestamp.utcnow()}])
    output_path = Path(output_path)

    if output_path.exists():
        existing = load_table(output_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    save_table(combined, output_path)
    return combined


def evaluate_churn_batch(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    y_true_col: str = "Churn",
    y_proba_col: str = "churn_probability",
    action_col: str | None = "action",
    threshold: float = 0.5,
) -> dict[str, Any]:
    df = load_table(input_path)

    metrics = compute_classification_metrics(
        df,
        y_true_col=y_true_col,
        y_proba_col=y_proba_col,
        threshold=threshold,
    )

    if action_col and action_col in df.columns:
        business_cfg = get_business_settings()

        business_metrics = compute_business_metrics(
            df,
            y_true_col=y_true_col,
            y_proba_col=y_proba_col,
            action_col=action_col,
            customer_value=business_cfg["customer_value"],
            cost_contact=business_cfg["cost_contact"],
            cost_discount=business_cfg["cost_discount"],
            contact_uplift=business_cfg["contact_uplift"],
            discount_uplift=business_cfg["discount_uplift"],
        )
        metrics.update({f"business_{k}": v for k, v in business_metrics.items()})

    if output_path is not None:
        append_metrics_history(metrics, output_path)

    return metrics


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate churn monitoring batch.")
    parser.add_argument("--input", required=True, help="CSV/parquet with labels + predictions.")
    parser.add_argument("--output", required=False, help="Metrics history output path.")
    parser.add_argument("--target-col", default="Churn")
    parser.add_argument("--proba-col", default="churn_probability")
    parser.add_argument("--action-col", default="action")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    metrics = evaluate_churn_batch(
        input_path=args.input,
        output_path=args.output,
        y_true_col=args.target_col,
        y_proba_col=args.proba_col,
        action_col=args.action_col,
        threshold=args.threshold,
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()