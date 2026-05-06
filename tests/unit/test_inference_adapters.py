import pandas as pd
import pytest

from src.inference.adapters import request_to_dataframe


def test_request_to_dataframe_success():
    inputs = [
        {
            "customerID": "1234-ABCDE",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": "845.50",
        },
        {
            "customerID": "5678-FGHIJ",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 45.0,
            "TotalCharges": "135.0",
        },
    ]

    df = request_to_dataframe(inputs)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == list(inputs[0].keys())
    assert df.loc[0, "customerID"] == "1234-ABCDE"
    assert df.loc[1, "tenure"] == 3


def test_request_to_dataframe_empty():
    with pytest.raises(ValueError, match="No input rows provided"):
        request_to_dataframe([])