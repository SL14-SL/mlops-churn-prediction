import os
import json
import socket
import traceback
import time
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
        mlflow.set_tracking_uri(resolve_tracking_uri())
        logger.info(f"Loading champion model: {MODEL_NAME}")

        try:
            (
                model,
                model_type,
                serving_alias,
                model_uri,
            ) = load_registry_model(MODEL_NAME)

            if serving_alias and serving_alias != "unknown":
                client = MlflowClient()
                version = client.get_model_version_by_alias(MODEL_NAME, serving_alias)
                serving_model_version = str(version.version)
                serving_model_run_id = version.run_id

            logger.info(f"✅ Model loaded: {MODEL_NAME} (Version: {serving_model_version})")
        except Exception as model_err:
            logger.error(f"❌ Failed to load model from registry: {model_err}")
            model = None

        yield
    finally:
        logger.info("Shutdown: Cleaning up resources.")

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
            model_type=model_type
        )
        timings["alignment"] = _ms_since(t)
        cat_base_cols = TRAIN_CFG.get("features", {}).get("categorical_columns", [])

        for col in final_df.columns:
            # Eine Spalte ist eine OHE-Spalte, wenn sie mit einem Kategorienamen beginnt
            is_ohe_col = any(col.startswith(f"{base}_") for base in cat_base_cols)
            
            if is_ohe_col:
                # Nur diese Spalten werden zu bool
                final_df[col] = final_df[col].astype(bool)
            elif col in ["tenure", "monthlycharges", "totalcharges"]:
                # Diese Spalten müssen explizit numerisch bleiben
                final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

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