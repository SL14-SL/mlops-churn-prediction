from __future__ import annotations

from typing import Any
import pandas as pd

def request_to_dataframe(inputs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert generic API inputs into a pandas DataFrame.
    This works for any problem type (Churn, Forecasting, etc.).
    """
    input_df = pd.DataFrame(inputs)
    
    if input_df.empty:
        raise ValueError("No input rows provided")
        
    return input_df