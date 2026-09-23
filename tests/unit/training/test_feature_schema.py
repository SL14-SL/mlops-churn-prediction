import json

import pandas as pd

from mlops_churn_prediction.training.feature_schema import (
    build_feature_schema,
    save_feature_schema,
)


def test_builds_feature_schema() -> None:
    dataframe = pd.DataFrame(
        {
            "tenure": pd.Series(
                [1, 2],
                dtype="int64",
            ),
            "contract": pd.Series(
                ["monthly", "annual"],
                dtype="category",
            ),
        }
    )

    schema = build_feature_schema(
        dataframe
    )

    assert schema["columns"] == [
        "tenure",
        "contract",
    ]
    assert schema["dtypes"] == {
        "tenure": "int64",
        "contract": "category",
    }


def test_saves_feature_schema(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "tenure": [1, 2],
        }
    )
    schema_path = (
        tmp_path / "feature_schema.json"
    )

    schema = save_feature_schema(
        dataframe,
        str(schema_path),
    )

    stored_schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored_schema == schema