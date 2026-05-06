import pandas as pd
import pytest

from src.monitoring.performance import compute_classification_metrics


def test_compute_classification_metrics_for_churn():
    df = pd.DataFrame(
        {
            "Churn": ["No", "Yes", "Yes", "No"],
            "churn_probability": [0.1, 0.8, 0.7, 0.3],
        }
    )

    metrics = compute_classification_metrics(df, threshold=0.5)

    assert metrics["n_samples"] == 4
    assert metrics["threshold"] == 0.5
    assert metrics["true_negatives"] == 2
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["predicted_churn_rate"] == 0.5
    assert metrics["actual_churn_rate"] == 0.5


def test_compute_classification_metrics_missing_required_columns():
    df = pd.DataFrame(
        {
            "Churn": ["No", "Yes"],
        }
    )

    with pytest.raises(KeyError, match="Missing required columns"):
        compute_classification_metrics(df)


def test_compute_classification_metrics_empty_after_dropna():
    df = pd.DataFrame(
        {
            "Churn": [None],
            "churn_probability": [None],
        }
    )

    with pytest.raises(ValueError, match="No rows available"):
        compute_classification_metrics(df)