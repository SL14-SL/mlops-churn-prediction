import pandera.pandas as pa
import pandas as pd
import pytest

from src.data.validation.validate import validate_train


@pytest.fixture
def valid_churn_df():
    return pd.DataFrame(
        {
            "customerID": ["1234-ABCDE"],
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["Fiber optic"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.35],
            "TotalCharges": ["845.50"],
            "Churn": ["Yes"],
        }
    )


def test_validate_train_happy_path(valid_churn_df):
    result_df = validate_train(valid_churn_df)

    assert not result_df.empty
    assert "customerID" in result_df.columns
    assert "Churn" in result_df.columns


@pytest.mark.parametrize(
    "invalid_col, bad_value",
    [
        ("gender", "Other"),
        ("SeniorCitizen", 2),
        ("Partner", "Maybe"),
        ("Dependents", "Maybe"),
        ("tenure", -1),
        ("PhoneService", "Maybe"),
        ("MultipleLines", "Maybe"),
        ("InternetService", "Cable"),
        ("Contract", "Weekly"),
        ("PaperlessBilling", "Maybe"),
        ("MonthlyCharges", -1.0),
        ("Churn", "Maybe"),
    ],
)
def test_validate_train_logic_errors(valid_churn_df, invalid_col, bad_value):
    df = valid_churn_df.copy()
    df[invalid_col] = [bad_value]

    with pytest.raises(pa.errors.SchemaError):
        validate_train(df)


def test_validate_train_missing_required_column(valid_churn_df):
    df = valid_churn_df.drop(columns=["customerID"])

    with pytest.raises(pa.errors.SchemaError):
        validate_train(df)


def test_validate_train_allows_missing_optional_target_for_inference_like_data(valid_churn_df):
    df = valid_churn_df.drop(columns=["Churn"])

    result_df = validate_train(df)

    assert not result_df.empty
    assert "Churn" not in result_df.columns


def test_validate_train_coerces_numeric_columns(valid_churn_df):
    df = valid_churn_df.copy()
    df["SeniorCitizen"] = ["1"]
    df["tenure"] = ["12"]
    df["MonthlyCharges"] = ["70.35"]

    result_df = validate_train(df)

    assert result_df["SeniorCitizen"].iloc[0] == 1
    assert result_df["tenure"].iloc[0] == 12
    assert result_df["MonthlyCharges"].iloc[0] == 70.35