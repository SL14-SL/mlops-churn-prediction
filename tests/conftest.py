# ruff: noqa: E402
import os

TEST_API_KEY = "test-secret-key"

os.environ["API_KEY"] = TEST_API_KEY
os.environ["PREFECT_API_MODE"] = "ephemeral"
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "true"

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.app import app

@pytest.fixture
def sample_churn_customer():
    return {
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
    }


@pytest.fixture
def sample_prediction_payload(sample_churn_customer):
    return {
        "inputs": [sample_churn_customer],
        "context": {"request_id": "test-request"},
    }


@pytest.fixture
def sample_prediction_df(sample_churn_customer):
    return pd.DataFrame([sample_churn_customer])

@pytest.fixture
def mock_xgb_model():
    model = MagicMock()

    model.get_booster.return_value.feature_names = [
        "tenure",
        "monthlycharges",
        "totalcharges",
        "seniorcitizen",
        "gender_Male",
        "partner_Yes",
        "dependents_Yes",
        "phoneservice_Yes",
        "multiplelines_Yes",
        "internetservice_Fiber optic",
        "internetservice_No",
        "onlinesecurity_Yes",
        "onlinebackup_Yes",
        "deviceprotection_Yes",
        "techsupport_Yes",
        "streamingtv_Yes",
        "streamingmovies_Yes",
        "contract_One year",
        "contract_Two year",
        "paperlessbilling_Yes",
        "paymentmethod_Credit card (automatic)",
        "paymentmethod_Electronic check",
        "paymentmethod_Mailed check",
    ]

    model.predict.return_value = [0.82]

    return model

@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def api_headers():
    return {"X-API-KEY": TEST_API_KEY}