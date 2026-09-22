import json
from typing import Any

import pandas as pd

from mlops_churn_prediction.storage.filesystem import (
    write_text,
)


def build_feature_schema(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Build the serving schema for a model feature table."""
    return {
        "columns": list(dataframe.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype
            in dataframe.dtypes.items()
        },
    }


def save_feature_schema(
    dataframe: pd.DataFrame,
    path: str,
) -> dict[str, Any]:
    """Persist a feature schema to local or GCS storage."""
    schema = build_feature_schema(
        dataframe
    )

    write_text(
        path,
        json.dumps(
            schema,
            indent=2,
        ),
    )

    return schema