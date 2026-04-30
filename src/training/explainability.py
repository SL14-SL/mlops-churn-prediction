import shap
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from src.utils.logger import get_logger
import pandas as pd
from collections import defaultdict

from src.configs.loader import get_path


logger = get_logger(__name__)

def log_feature_importance(model, feature_names: list[str]) -> None:
    """
    Log feature importance table and plot to MLflow.

    Helps explain which customer attributes drive churn predictions.
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

    output_dir = Path(get_path("models")) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "feature_importance.csv"
    plot_path = output_dir / "feature_importance_top20.png"

    importance_df.to_csv(csv_path, index=False)

    top_n = importance_df.head(20).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_n["feature"], top_n["importance"])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(str(csv_path), artifact_path="feature_importance")
    mlflow.log_artifact(str(plot_path), artifact_path="feature_importance")

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
        elif pd.api.types.is_categorical_dtype(shap_df[col]):
            shap_df[col] = shap_df[col].cat.codes
        else:
            shap_df[col] = pd.to_numeric(shap_df[col], errors="coerce")

    return shap_df.fillna(0.0).astype(float)


def log_shap_summary(model, X_sample: pd.DataFrame):
    """
    Log SHAP summary plot to MLflow.
    Shows global feature impact.
    """

    logger.info("Calculating SHAP values...")

    X_sample = prepare_shap_input(X_sample)

    X_sample_numeric = prepare_shap_input(X_sample)

    explainer = shap.Explainer(model, X_sample_numeric)
    shap_values = explainer(X_sample_numeric)

    # 👉 OHE zurück zu echten Features
    feature_map = build_ohe_feature_map(X_sample.columns)

    shap_agg = aggregate_ohe_shap_values(
        shap_values.values,
        X_sample,
        feature_map
    )

    plt.figure()
    shap.summary_plot(shap_agg.values, shap_agg, show=False)

    plot_path = Path(get_path("models")) / "reports" / "shap_summary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(plot_path), artifact_path="shap")

    logger.info("SHAP summary logged to MLflow.")

def aggregate_ohe_shap_values(shap_values, X, original_feature_map):
    """
    Aggregate SHAP values from OHE features back to original features.

    original_feature_map:
        {
            "internetservice": ["internetservice_Fiber optic", "internetservice_No"],
            ...
        }
    """

    shap_df = pd.DataFrame(shap_values, columns=X.columns)

    aggregated = {}

    for original_feature, ohe_cols in original_feature_map.items():
        existing_cols = [c for c in ohe_cols if c in shap_df.columns]
        if existing_cols:
            aggregated[original_feature] = shap_df[existing_cols].sum(axis=1)

    return pd.DataFrame(aggregated)

def build_ohe_feature_map(columns):
    """
    Detect OHE groups from column names.
    """

    mapping = defaultdict(list)

    for col in columns:
        if "_" in col:
            base = col.split("_")[0]
            mapping[base].append(col)

    return dict(mapping)