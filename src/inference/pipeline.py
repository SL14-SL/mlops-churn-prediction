import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.inference.decision import DecisionConfig, DecisionEngine
from src.configs.loader import load_config

logger = get_logger(__name__)

CFG = load_config()

decision_engine = DecisionEngine(
    DecisionConfig.from_config(CFG)
)

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
) -> pd.DataFrame:
    """
    Aligns inference dataframe to the exact feature order expected by the model.
    Fills missing One-Hot columns with 0.
    """
    try:
        # 1. Get the list of features the model was trained on
        if model_type == "xgboost":
            # For XGBoost (Booster/Pyfunc)
            try:
                # Try getting it from the booster directly
                raw_features = model.get_booster().feature_names
            except AttributeError:
                # If it's wrapped in an MLflow Pyfunc object
                raw_features = model.metadata.get_input_schema().input_names()
        else:
            # Fallback for other model types via MLflow Signature
            raw_features = model.metadata.get_input_schema().input_names()

        model_features = [f.lower() for f in raw_features]

        # 2. Critical Step: Fill missing columns with 0
        # If the model wants 'gender_male' but we only sent a Female record,
        # we create 'gender_male' and set it to 0.
        missing_cols = set(model_features) - set(processed_df.columns)
        for col in missing_cols:
            processed_df[col] = 0

        # 3. Align order and return
        return processed_df[model_features]

    except Exception as e:
        logger.error(f"Feature alignment failed: {e}")
        # Show which columns were expected vs which were there for better debugging
        if 'model_features' in locals():
             logger.debug(f"Expected: {model_features}")
             logger.debug(f"Available: {list(processed_df.columns)}")
        raise

def predict_and_decide(
    input_df: pd.DataFrame,
    model,
    model_type: str,
) -> list[dict]:

    # 1. validate
    validated_df = validate_prediction_input(input_df)

    # 2. align
    aligned_df = align_features_for_model(
        processed_df=validated_df,
        model=model,
        model_type=model_type,
    )

    # 3. predict
    try:
        if model_type == "xgboost":
            probs = model.predict(aligned_df)
        else:
            probs = model.predict_proba(aligned_df)[:, 1]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

    probs = np.array(probs).flatten()
    probs = apply_prediction_postprocessing(probs)

    # 4. decision
    results = [decision_engine.decide(p) for p in probs]

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