import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
from src.data.validation.churn_schema import ChurnSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main validation function for churn data (train and batch).
    """
    try:
        # Pandera performs type checking and value range validation
        validated_df = ChurnSchema.validate(df)
        logger.info("Churn data validation successful.")
        return validated_df
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

def validate_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data incoming via API for prediction.
    Target column 'Churn' should be absent here.
    """
    # Create a copy of the schema but without the target column for inference
    inference_schema = ChurnSchema.to_schema().remove_columns(["Churn"])
    
    try:
        validated = inference_schema.validate(df)
        if validated.empty:
            raise ValueError("Inference input is empty.")
        return validated
    except Exception as e:
        logger.error(f"Inference validation failed: {e}")
        raise

