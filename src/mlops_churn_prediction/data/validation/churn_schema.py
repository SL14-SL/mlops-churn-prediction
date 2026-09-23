import pandera.pandas as pa
from pandera.typing import Series
from typing import Optional

class ChurnSchema(pa.DataFrameModel):
    """
    Schema for churn prediction data.
    """
    customerID: Series[str] = pa.Field(nullable=False)
    gender: Series[str] = pa.Field(isin=["Female", "Male"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(isin=["Yes", "No", "No phone service"])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(nullable=True)
    OnlineBackup: Series[str] = pa.Field(nullable=True)
    DeviceProtection: Series[str] = pa.Field(nullable=True)
    TechSupport: Series[str] = pa.Field(nullable=True)
    StreamingTV: Series[str] = pa.Field(nullable=True)
    StreamingMovies: Series[str] = pa.Field(nullable=True)
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(nullable=True)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    
    # TotalCharges contains empty strings " " in raw data. 
    TotalCharges: Optional[Series[str]] = pa.Field(nullable=True)
    
    # Target column is optional because it's missing during inference (API calls)
    Churn: Optional[Series[str]] = pa.Field(isin=["Yes", "No"], nullable=True)

    class Config:
        strict = False
        coerce = True