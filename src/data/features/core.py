import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def cast_numeric_types(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts specified columns to numeric, handling common data issues
    like empty strings " " being present in numeric columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # Errors='coerce' replaces " " with NaN, then we fill with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Transforms categorical string columns into numerical dummies.
    """
    df = df.copy()
    # Using drop_first=True to reduce multicollinearity
    # (Especially important for linear models, but good practice for GB too)
    return pd.get_dummies(df, columns=columns, drop_first=True)


def drop_unnecessary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes columns that should not be used as features (IDs, raw targets).
    """
    df = df.copy()
    existing_cols = [c for c in columns if c in df.columns]
    return df.drop(columns=existing_cols)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names for the model: lowercase, no spaces, no special characters.
    """
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") 
        for col in df.columns
    ]
    return df