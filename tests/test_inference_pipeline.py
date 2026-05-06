import pandas as pd

from src.inference.pipeline import (
    validate_prediction_input,
    align_features_for_model,
)


def test_validate_prediction_input_normalizes_churn_columns(sample_prediction_df):
    result = validate_prediction_input(sample_prediction_df)

    assert "customerid" in result.columns
    assert "tenure" in result.columns
    assert "monthlycharges" in result.columns
    assert "totalcharges" in result.columns


def test_validate_prediction_input_preserves_row_count(sample_prediction_df):
    result = validate_prediction_input(sample_prediction_df)

    assert len(result) == len(sample_prediction_df)


def test_validate_prediction_input_normalizes_target_column_name(sample_prediction_df):
    df = sample_prediction_df.copy()
    df["Churn"] = "Yes"

    result = validate_prediction_input(df)

    assert "churn" in result.columns


def test_align_features_for_churn_xgboost(mock_xgb_model):
    processed_df = pd.DataFrame(
        [
            {
                "tenure": 12,
                "monthlycharges": 70.35,
                "totalcharges": 845.50,
                "seniorcitizen": 0,
                "gender_Male": False,
                "partner_Yes": True,
                "extra_column": 999,
            }
        ]
    )

    result = align_features_for_model(
        processed_df=processed_df,
        model=mock_xgb_model,
        model_type="xgboost",
        feature_schema=None,
    )

    assert list(result.columns) == mock_xgb_model.get_booster.return_value.feature_names
    assert "extra_column" not in result.columns


def test_align_features_adds_missing_churn_features(mock_xgb_model):
    processed_df = pd.DataFrame(
        [
            {
                "tenure": 12,
            }
        ]
    )

    result = align_features_for_model(
        processed_df=processed_df,
        model=mock_xgb_model,
        model_type="xgboost",
        feature_schema=None,
    )

    assert list(result.columns) == mock_xgb_model.get_booster.return_value.feature_names
    assert result["tenure"].iloc[0] == 12
    assert result["monthlycharges"].iloc[0] == 0
    assert result["totalcharges"].iloc[0] == 0