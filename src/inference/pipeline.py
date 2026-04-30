import pandas as pd
import numpy as np
import mlflow

from src.utils.logger import get_logger
from src.inference.decision import DecisionConfig, DecisionEngine
from src.configs.loader import load_config

logger = get_logger(__name__)

CFG = load_config()


def validate_prediction_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize raw prediction input.
    Ensures column names are consistent (lowercase, underscores).
    """
    # 1. Basic Cleaning (match training preprocessing)
    validated_df = input_df.copy()
    validated_df.columns = [
        str(col).lower().replace(" ", "_") 
        for col in validated_df.columns
    ]
    
    return validated_df


def align_features_for_model(
    processed_df: pd.DataFrame,
    model,
    model_type: str,
    feature_schema: dict | None = None,
) -> pd.DataFrame:
    """
    Align inference dataframe to the exact feature structure expected by the model.

    Priority:
    1. feature_schema (preferred, explicit)
    2. fallback to model metadata (MLflow signature)

    Handles:
    - missing columns (filled with 0)
    - column order
    - dtype enforcement
    """
    try:
        df = processed_df.copy()

        # --- 1. Resolve expected feature list ---
        if feature_schema:
            model_features = feature_schema["columns"]
            expected_dtypes = feature_schema.get("dtypes", {})
        else:
            # Fallback to MLflow metadata
            if model_type == "xgboost":
                try:
                    raw_features = model.get_booster().feature_names
                except AttributeError:
                    raw_features = model.metadata.get_input_schema().input_names()
            else:
                raw_features = model.metadata.get_input_schema().input_names()

            model_features = [f.lower() for f in raw_features]
            expected_dtypes = {}

        # --- 2. Normalize column names ---
        df.columns = [str(col).lower() for col in df.columns]

        # --- 3. Add missing columns ---
        missing_cols = set(model_features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        if missing_cols:
            logger.warning(f"Missing features filled with 0: {missing_cols}")

        # --- 4. Remove unexpected columns ---
        extra_cols = set(df.columns) - set(model_features)
        if extra_cols:
            logger.warning(f"Dropping unexpected columns: {extra_cols}")
            df = df.drop(columns=list(extra_cols))

        # --- 5. Enforce column order ---
        df = df[model_features]

        # --- 6. Enforce dtypes (if schema available) ---
        for col, dtype in expected_dtypes.items():
            if col in df:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    logger.warning(f"Failed to cast column {col} to {dtype}")

        return df

    except Exception as e:
        logger.error(f"Feature alignment failed: {e}")
        logger.debug(f"Available columns: {list(processed_df.columns)}")
        raise

def _build_decision_engine():
    """
    Build decision engine using:
    1. Base config (dev/prod.yaml)
    2. Override with model-specific threshold (if available)
    """
    decision_config = DecisionConfig.from_config(CFG)

    return DecisionEngine(decision_config)


def predict_and_decide(input_df: pd.DataFrame, model) -> list[dict]:

    decision_engine = _build_decision_engine()

    # 1. Predict probabilities
    raw_preds = model.predict(input_df)

    # Handle pyfunc output
    if isinstance(raw_preds, list) and isinstance(raw_preds[0], dict):
        probs = raw_preds[0]["probabilities"]
    else:
        probs = raw_preds

    probs = np.asarray(probs).reshape(-1)   # <-- statt nur flatten
    probs = probs.astype(float)

    # 2. Decision
    # results = [decision_engine.decide(p) for p in probs]
    results = [decision_engine.decide_batch(probs)]
    if isinstance(results, list) and len(results) == 1 and isinstance(results[0], list):
        results = results[0]
        
    logger.info(f"DEBUG decide_batch output: {results}")
    logger.info(f"DEBUG probs shape: {np.array(raw_preds).shape}")


    return results

def apply_prediction_postprocessing(
    predictions: list[float],
) -> list[float]:
    """
    Apply optional domain-specific post-processing for classification.
    Currently, we just ensure the output format is consistent.
    """
    # might want to clip values or apply a threshold
    # But for now,keep the raw model output (0/1 or probability)
    return [float(pred) for pred in predictions]