import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

def cast_object_columns_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns to pandas category dtype for model compatibility.
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        logger.info(f"Converting categorical columns: {cat_cols}")
        for col in cat_cols:
            df[col] = df[col].astype(str).astype("category")

    return df


def drop_columns_if_present(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns if they exist in the dataframe.
    """
    df = df.copy()

    existing_drops = [col for col in columns if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    return df