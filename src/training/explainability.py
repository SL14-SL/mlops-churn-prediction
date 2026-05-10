from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap

from src.configs.loader import ensure_dir, get_path
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _reports_path(filename: str) -> str:
    """
    Build the configured reports path for local or remote storage.
    """
    return f"{get_path('models')}/reports/{filename}"


def log_feature_importance(model, feature_names: list[str]) -> None:
    """
    Log feature importance table and plot to MLflow.

    The permanent report paths can point to local storage or GCS. Temporary
    local files are still used for MLflow artifact logging because MLflow
    expects local file paths for `log_artifact`.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_. Skipping.")
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    reports_dir = f"{get_path('models')}/reports"
    ensure_dir(reports_dir)

    csv_path = _reports_path("feature_importance.csv")
    plot_path = _reports_path("feature_importance_top20.png")

    importance_df.to_csv(csv_path, index=False)

    top_n = importance_df.head(20).sort_values("importance")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_csv_path = Path(tmpdir) / "feature_importance.csv"
        local_plot_path = Path(tmpdir) / "feature_importance_top20.png"

        importance_df.to_csv(local_csv_path, index=False)

        plt.figure(figsize=(10, 8))
        plt.barh(top_n["feature"], top_n["importance"])
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(local_plot_path)
        plt.close()

        pd.DataFrame(top_n).to_csv(csv_path, index=False)

        # Keep the configured reports output as before, but log from local temp
        # files because MLflow requires local artifact paths.
        if csv_path != str(local_csv_path):
            importance_df.to_csv(csv_path, index=False)

        if plot_path != str(local_plot_path):
            # pandas/fsspec handles CSV directly, but matplotlib cannot save
            # reliably to all remote filesystems. Therefore, persist the plot
            # via pandas-compatible local MLflow artifact only.
            pass

        mlflow.log_artifact(str(local_csv_path), artifact_path="feature_importance")
        mlflow.log_artifact(str(local_plot_path), artifact_path="feature_importance")

    logger.info("Feature importance logged to MLflow.")


def prepare_shap_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert training features into a numeric SHAP-compatible dataframe.

    TreeExplainer requires numeric input without object/category dtypes.
    """
    shap_df = df.copy()

    for col in shap_df.columns:
        if pd.api.types.is_bool_dtype(shap_df[col]):
            shap_df[col] = shap_df[col].astype(int)
        elif isinstance(shap_df[col].dtype, pd.CategoricalDtype):
            shap_df[col] = shap_df[col].cat.codes
        else:
            shap_df[col] = pd.to_numeric(shap_df[col], errors="coerce")

    return shap_df.fillna(0.0).astype(float)


def log_shap_summary(model, X_sample: pd.DataFrame) -> None:
    """
    Log SHAP summary plot to MLflow.

    Shows global feature impact. A temporary local image is used for MLflow
    artifact logging.
    """
    logger.info("Calculating SHAP values...")

    X_sample = prepare_shap_input(X_sample)
    X_sample_numeric = prepare_shap_input(X_sample)

    explainer = shap.Explainer(model, X_sample_numeric)
    shap_values = explainer(X_sample_numeric)

    feature_map = build_ohe_feature_map(X_sample.columns)

    shap_agg = aggregate_ohe_shap_values(
        shap_values.values,
        X_sample,
        feature_map,
    )

    reports_dir = f"{get_path('models')}/reports"
    ensure_dir(reports_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_plot_path = Path(tmpdir) / "shap_summary.png"

        plt.figure()
        shap.summary_plot(shap_agg.values, shap_agg, show=False)
        plt.savefig(local_plot_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(local_plot_path), artifact_path="shap")

    logger.info("SHAP summary logged to MLflow.")


def aggregate_ohe_shap_values(
    shap_values,
    X: pd.DataFrame,
    original_feature_map: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Aggregate SHAP values from OHE features back to original features.

    Example:
        {
            "internetservice": [
                "internetservice_Fiber optic",
                "internetservice_No",
            ],
        }
    """
    shap_df = pd.DataFrame(shap_values, columns=X.columns)

    aggregated = {}

    for original_feature, ohe_cols in original_feature_map.items():
        existing_cols = [c for c in ohe_cols if c in shap_df.columns]
        if existing_cols:
            aggregated[original_feature] = shap_df[existing_cols].sum(axis=1)

    return pd.DataFrame(aggregated)


def build_ohe_feature_map(columns) -> dict[str, list[str]]:
    """
    Detect one-hot encoded feature groups from column names.
    """
    mapping = defaultdict(list)

    for col in columns:
        if "_" in col:
            base = col.split("_")[0]
            mapping[base].append(col)

    return dict(mapping)