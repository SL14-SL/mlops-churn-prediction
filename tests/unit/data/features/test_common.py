import pandas as pd

from src.data.features.common import (
    cast_object_columns_to_category,
    drop_columns_if_present,
)


def test_cast_object_columns_to_category_converts_churn_categoricals():
    df = pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "contract": ["Month-to-month", "Two year"],
            "paymentmethod": ["Electronic check", "Mailed check"],
            "tenure": [12, 24],
            "monthlycharges": [70.35, 45.00],
        }
    )

    result = cast_object_columns_to_category(df)

    assert str(result["gender"].dtype) == "category"
    assert str(result["contract"].dtype) == "category"
    assert str(result["paymentmethod"].dtype) == "category"
    assert result["tenure"].dtype == df["tenure"].dtype
    assert result["monthlycharges"].dtype == df["monthlycharges"].dtype


def test_drop_columns_if_present_drops_existing_columns_only():
    df = pd.DataFrame(
        {
            "customerid": ["1234-ABCDE"],
            "tenure": [12],
            "monthlycharges": [70.35],
            "churn": [1],
        }
    )

    result = drop_columns_if_present(df, ["customerid", "missing_column"])

    assert "customerid" not in result.columns
    assert "tenure" in result.columns
    assert "monthlycharges" in result.columns
    assert "churn" in result.columns


def test_drop_columns_if_present_does_not_mutate_input():
    df = pd.DataFrame(
        {
            "customerid": ["1234-ABCDE"],
            "tenure": [12],
        }
    )

    result = drop_columns_if_present(df, ["customerid"])

    assert "customerid" in df.columns
    assert "customerid" not in result.columns