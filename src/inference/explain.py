from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.configs.loader import get_path
from src.inference.pipeline import align_features_for_model
from src.training.explainability import prepare_shap_input


def cast_to_feature_schema(df: pd.DataFrame, schema: dict | None) -> pd.DataFrame:
    """
    Cast dataframe columns back to the MLflow model schema.

    Needed because SHAP uses numeric arrays, while MLflow pyfunc enforces
    the original input schema.
    """
    if schema is None:
        return df

    df = df.copy()

    for col, dtype in schema.get("dtypes", {}).items():
        if col not in df.columns:
            continue

        if dtype == "bool":
            df[col] = df[col].astype(bool)
        elif str(dtype).startswith("int"):
            df[col] = df[col].round().astype(dtype)
        elif str(dtype).startswith("float"):
            df[col] = df[col].astype(dtype)

    return df


def load_shap_background(
    *,
    model: Any,
    model_type: str,
    feature_schema: dict | None,
    train_cfg: dict,
    sample_size: int = 200,
) -> pd.DataFrame:
    """
    Load a small feature background dataset for SHAP explanations.

    Uses training split features as the reference population. The path is kept
    as a string so pandas can read from local storage or GCS.
    """
    train_path = f"{get_path('splits')}/train.parquet"
    df = pd.read_parquet(train_path)

    target_col = train_cfg["data"]["target_column"]
    drop_cols = [target_col] + train_cfg.get("features", {}).get("drop_columns", [])

    X = df.drop(columns=drop_cols, errors="ignore")

    X = align_features_for_model(
        processed_df=X,
        model=model,
        model_type=model_type,
        feature_schema=feature_schema,
    )

    sample_n = min(sample_size, len(X))
    return prepare_shap_input(X.sample(sample_n, random_state=42))


def explain_single_prediction(
    *,
    final_df: pd.DataFrame,
    model: Any,
    model_type: str,
    feature_schema: dict | None,
    train_cfg: dict,
    top_n: int = 5,
) -> list[dict]:
    """
    Explain a single prediction using SHAP values.

    Returns the top features contributing to the model output.
    """
    if len(final_df) != 1:
        raise ValueError("Explanation currently supports exactly one row.")

    shap_input = prepare_shap_input(final_df)

    background = load_shap_background(
        model=model,
        model_type=model_type,
        feature_schema=feature_schema,
        train_cfg=train_cfg,
    )

    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=shap_input.columns)
        x_df = cast_to_feature_schema(x_df, feature_schema)
        preds = model.predict(x_df)
        return np.asarray(preds).reshape(-1)

    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(shap_input)

    values = shap_values.values[0]

    explanation_df = pd.DataFrame(
        {
            "feature": shap_input.columns,
            "impact": values,
            "abs_impact": np.abs(values),
            "value": shap_input.iloc[0].values,
        }
    )

    explanation_df = explanation_df[explanation_df["abs_impact"] > 1e-6].copy()

    if explanation_df.empty:
        return []

    explanation_df["direction"] = np.where(
        explanation_df["impact"] > 0,
        "increases_churn",
        "reduces_churn",
    )

    total_abs_impact = explanation_df["abs_impact"].sum()
    explanation_df["impact_pct"] = (
        explanation_df["abs_impact"] / total_abs_impact * 100
    ).round(2)

    explanation_df = explanation_df.sort_values("abs_impact", ascending=False)

    return (
        explanation_df.head(top_n)
        .drop(columns=["abs_impact"])
        .to_dict(orient="records")
    )