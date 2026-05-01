import os
import json
import socket
import traceback
import time
import shap
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from mlflow import MlflowClient

import mlflow
from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.status import HTTP_403_FORBIDDEN

from src.api.schema import PredictionRequest, PredictionResponse
from src.configs.loader import load_config, get_path
from src.inference.decision import DecisionEngine, DecisionConfig

from src.monitoring.prediction_logger import log_prediction
from src.monitoring.data_quality import (
    initialize_data_quality_reference_cache, 
    build_reference_category_cache, 
    log_data_quality_runtime
)
from src.monitoring.config import get_serving_settings, get_data_quality_settings
from src.monitoring.serving import normalize_path, observe_request, should_ignore_path

from src.data.features.build_features import build_features

from src.inference.pipeline import (
    validate_prediction_input, 
    align_features_for_model,
    predict_and_decide,
)
from src.inference.adapters import request_to_dataframe
from src.inference.router import load_registry_model
from src.inference.schema import load_feature_schema
from src.training.explainability import prepare_shap_input

from src.utils.logger import get_logger

logger = get_logger(__name__)

def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)

# --- 1. Load configuration and paths ---
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]
MODELS_PATH = Path(get_path("models"))

# Global variables for caching
model = None
model_type = "unknown"
serving_alias = "unknown"
model_uri = None
dq_reference_categories: dict[str, set[str]] = {}
serving_model_version = None
serving_model_run_id = None
feature_schema = None

# API Key Security Configuration
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the API Key from the request header."""
    expected_key = os.getenv("API_KEY")
    logger.info(f"DEBUG: Received: {api_key_header}, Expected: {expected_key}")
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )

def resolve_tracking_uri() -> str:
    """Determines the MLflow tracking URI based on the environment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        is_docker = os.path.exists("/.dockerenv")
        if is_docker:
            try:
                mlflow_ip = socket.gethostbyname("mlflow")
                tracking_uri = f"http://{mlflow_ip}:5000"
            except Exception:
                tracking_uri = "http://mlflow:5000"
        else:
            tracking_uri = CFG.get("mlflow_tracking_uri", "http://localhost:5000")
    return tracking_uri

def reload_serving_model() -> dict:
    """
    Reload the current champion model from the MLflow Model Registry.

    Updates the in-memory model used by the API without restarting the service.
    """
    global model, model_type, serving_alias, model_uri
    global serving_model_version, serving_model_run_id, feature_schema

    mlflow.set_tracking_uri(resolve_tracking_uri())

    (
        model,
        model_type,
        serving_alias,
        model_uri,
    ) = load_registry_model(MODEL_NAME)

    feature_schema = load_feature_schema()

    serving_model_version = None
    serving_model_run_id = None

    if serving_alias and serving_alias != "unknown":
        client = MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, serving_alias)
        serving_model_version = str(version.version)
        serving_model_run_id = version.run_id

    logger.info(
        f"Model reloaded: {MODEL_NAME} "
        f"(alias={serving_alias}, version={serving_model_version})"
    )

    return {
        "model_name": MODEL_NAME,
        "serving_alias": serving_alias,
        "model_version": serving_model_version,
        "model_run_id": serving_model_run_id,
        "model_uri": model_uri,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown. 
    Loads the ML model from registry and initializes data quality caches.
    """
    global model, model_type, model_uri, serving_alias
    global serving_model_version, serving_model_run_id, dq_reference_categories

    try:
        # --- Initialize Data Quality Cache ---
        ref_df = initialize_data_quality_reference_cache()
        dq_reference_categories = build_reference_category_cache(
            ref_df,
            categorical_reference_features=get_data_quality_settings().get(
                "categorical_reference_features", []
            ),
        )

        # --- Load Model from MLflow ---
        try:
            reload_serving_model()
            logger.info(f"✅ Model loaded: {MODEL_NAME} (Version: {serving_model_version})")
        except Exception as model_err:
            logger.error(f"❌ Failed to load model from registry: {model_err}")
            model = None

        yield
    finally:
        logger.info("Shutdown: Cleaning up resources.")

def cast_to_feature_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Cast dataframe columns back to the MLflow model schema.

    Needed because SHAP uses numeric arrays, while MLflow pyfunc enforces the original input schema.
    """
    df = df.copy()

    for col, dtype in schema.get("dtypes", {}).items():
        if col not in df.columns:
            continue

        if dtype == "bool":
            df[col] = df[col].astype(bool)
        elif dtype.startswith("int"):
            df[col] = df[col].round().astype(dtype)
        elif dtype.startswith("float"):
            df[col] = df[col].astype(dtype)

    return df

def load_shap_background() -> pd.DataFrame:
    """
    Load a small feature background dataset for SHAP explanations.

    Uses training split features as the reference population.
    """
    train_path = Path(get_path("splits")) / "train.parquet"
    df = pd.read_parquet(train_path)

    target_col = TRAIN_CFG["data"]["target_column"]
    drop_cols = [target_col] + TRAIN_CFG.get("features", {}).get("drop_columns", [])

    X = df.drop(columns=drop_cols, errors="ignore")
    X = align_features_for_model(
        processed_df=X,
        model=model,
        model_type=model_type,
        feature_schema=feature_schema,
    )

    return prepare_shap_input(X.sample(min(200, len(X)), random_state=42))


def explain_single_prediction(final_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Explain a single prediction using SHAP values.

    Returns the top features contributing to the model output.
    """
    if len(final_df) != 1:
        raise ValueError("Explanation currently supports exactly one row.")

    shap_input = prepare_shap_input(final_df)
    background = load_shap_background()

    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=shap_input.columns)
        x_df = cast_to_feature_schema(x_df, feature_schema)
        preds = model.predict(x_df)
        return np.asarray(preds).reshape(-1)

    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(shap_input)

    values = shap_values.values[0]

    explanation_df = pd.DataFrame(
        {
            "feature": shap_input.columns,
            "impact": values,
            "abs_impact": np.abs(values),
            "value": shap_input.iloc[0].values,
        }
    )

    explanation_df = explanation_df[explanation_df["abs_impact"] > 1e-6].copy()

    if explanation_df.empty:
        return []

    explanation_df["direction"] = np.where(
        explanation_df["impact"] > 0,
        "increases_churn",
        "reduces_churn",
    )

    total_abs_impact = explanation_df["abs_impact"].sum()
    explanation_df["impact_pct"] = (
        explanation_df["abs_impact"] / total_abs_impact * 100
    ).round(2)

    explanation_df = explanation_df.sort_values("abs_impact", ascending=False)

    return (
        explanation_df.head(top_n)
        .drop(columns=["abs_impact"])
        .to_dict(orient="records")
    )


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)
SERVING_CFG = get_serving_settings()

# --- Middleware for Monitoring & Prometheus ---
@app.middleware("http")
async def serving_monitoring_middleware(request: Request, call_next):
    if not SERVING_CFG.get("enabled", True):
        return await call_next(request)

    raw_path = request.url.path
    if should_ignore_path(raw_path, SERVING_CFG.get("ignored_paths")):
        return await call_next(request)

    method = request.method
    path = normalize_path(raw_path, SERVING_CFG.get("track_paths"))
    start = time.perf_counter()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        observe_request(
            method=method,
            path=path,
            status_code=status_code,
            latency_seconds=time.perf_counter() - start,
        )

@app.get("/metrics", include_in_schema=False)
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.post("/admin/reload-model")
def reload_model(api_key: str = Depends(get_api_key)):
    """
    Reload the current champion model from MLflow.

    Used after a new champion model version has been promoted.
    """
    try:
        result = reload_serving_model()
    except Exception as e:
        logger.error(f"Model reload failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

    return {
        "status": "reloaded",
        **result,
    }

@app.get("/health")
def health(response: Response):
    is_healthy = model is not None
    if not is_healthy:
        response.status_code = 503
    return {
        "status": "online" if is_healthy else "degraded",
        "model_name": MODEL_NAME,
        "serving_alias": serving_alias,
        "model_version": serving_model_version
    }

@app.post("/explain", dependencies=[Depends(get_api_key)])
def explain(payload: PredictionRequest, top_n: int = 5):
    """
    Return churn prediction with top feature-level explanation.

    Intended for debugging, demos, and customer-level model interpretation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    if len(payload.inputs) != 1:
        raise HTTPException(
            status_code=400,
            detail="Explain endpoint currently supports exactly one input row.",
        )

    try:
        input_df = request_to_dataframe(payload.inputs)
        validated_df = validate_prediction_input(input_df)
        processed_df = build_features(validated_df, config=TRAIN_CFG)

        final_df = align_features_for_model(
            processed_df=processed_df,
            model=model,
            model_type=model_type,
            feature_schema=feature_schema,
        )

        prediction = predict_and_decide(
            input_df=final_df,
            model=model,
        )[0]

        explanation = explain_single_prediction(final_df, top_n=top_n)

        return {
            "status": "success",
            "prediction": prediction,
            "top_reasons": explanation,
            "metadata": {
                "model_name": MODEL_NAME,
                "serving_alias": serving_alias,
                "model_version": serving_model_version,
            },
        }

    except Exception as e:
        logger.error(f"Explanation failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", dependencies=[Depends(get_api_key)], response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Main prediction endpoint. 
    Processes a batch of customer data and returns churn probabilities/classes.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    request_started = time.perf_counter()
    timings: dict[str, float] = {}
    request_id = payload.context.get("request_id", str(uuid4()))
    environment = os.getenv("APP_ENV", "dev")

    try:
        # 1. Convert input list to DataFrame
        t = time.perf_counter()
        input_df = request_to_dataframe(payload.inputs)
        timings["request_to_dataframe"] = _ms_since(t)

        # 2. Basic validation
        t = time.perf_counter()
        validated_df = validate_prediction_input(input_df)
        timings["validate_input"] = _ms_since(t)

        # 3. Data Quality Logging
        t = time.perf_counter()
        try:
            dq_summary = log_data_quality_runtime(
                validated_df, reference_categories=dq_reference_categories
            )
        except Exception as dq_error:
            dq_summary = {"status": "error", "message": str(dq_error)}
        timings["data_quality"] = _ms_since(t)

        # 4. Feature Engineering (Classification mode)
        t = time.perf_counter()
        processed_df = build_features(validated_df, config=TRAIN_CFG)
        # processed_df = validated_df

        timings["preprocessing"] = _ms_since(t)

        # 5. Feature Alignment (Ensure columns match model training)
        t = time.perf_counter()
        final_df = align_features_for_model(
            processed_df=processed_df, 
            model=model, 
            model_type=model_type,
            feature_schema=feature_schema,
        )
        timings["alignment"] = _ms_since(t)
        
        # 6. Model Inference (Batch prediction)
        t = time.perf_counter()

        results = predict_and_decide(
            input_df=final_df,
            model=model,
        )

        timings["inference"] = _ms_since(t)

        # 7. Async Logging & Response Construction
        timings["total"] = _ms_since(request_started)

        for features, result in zip(payload.inputs, results):
            log_prediction(
                features,
                result["churn_probability"],
                model_alias=serving_alias,
                model_version=serving_model_version,
                model_run_id=serving_model_run_id,
                request_id=request_id,
                environment=environment,
                action=result["action"],
                expected_value=result["expected_value"],
            )

        return {
            "predictions": results,
            "status": "success",
            "metadata": {
                "rows": len(results),
                "model_name": MODEL_NAME,
                "serving_alias": serving_alias,
                "request_id": request_id,
                "timing_ms": timings,
                "data_quality": dq_summary,
            },
        }

    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))