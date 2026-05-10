from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.configs.loader import ensure_dir, file_exists, get_path
from src.monitoring.config import get_data_quality_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MONITORING_PATH = get_path("monitoring")
VALIDATED_PATH = get_path("validated_data")

_REFERENCE_FRAME_CACHE: pd.DataFrame | None = None
_REFERENCE_CATEGORY_CACHE: dict[str, set[str]] = {}


def _history_path() -> str:
    """
    Return the data quality history path for local storage or GCS.
    """
    return f"{MONITORING_PATH}/data_quality_history.parquet"


def load_reference_frame() -> pd.DataFrame:
    """
    Load validated training data as reference baseline for category comparison.
    """
    path = f"{VALIDATED_PATH}/train.parquet"

    if not file_exists(path):
        logger.warning(f"Reference data not found for data quality: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)


def set_reference_frame_cache(ref_df: pd.DataFrame | None) -> None:
    """
    Set the in-memory reference dataframe cache.
    """
    global _REFERENCE_FRAME_CACHE
    _REFERENCE_FRAME_CACHE = ref_df.copy() if ref_df is not None else pd.DataFrame()


def get_reference_frame_cached() -> pd.DataFrame:
    """
    Return the cached reference dataframe, loading it if needed.
    """
    global _REFERENCE_FRAME_CACHE

    if _REFERENCE_FRAME_CACHE is None:
        _REFERENCE_FRAME_CACHE = load_reference_frame()

    return _REFERENCE_FRAME_CACHE.copy()


def build_reference_category_cache(
    ref_df: pd.DataFrame,
    categorical_reference_features: list[str],
) -> dict[str, set[str]]:
    """
    Build a mapping of known category values from the reference dataset.
    """
    cache: dict[str, set[str]] = {}

    if ref_df.empty:
        return cache

    for feature in categorical_reference_features:
        if feature not in ref_df.columns:
            logger.warning(
                f"Categorical reference feature missing in reference data: {feature}"
            )
            continue

        cache[feature] = set(ref_df[feature].dropna().astype(str).unique())

    return cache


def set_reference_category_cache(cache: dict[str, set[str]]) -> None:
    """
    Set the in-memory category reference cache.
    """
    global _REFERENCE_CATEGORY_CACHE
    _REFERENCE_CATEGORY_CACHE = cache.copy()


def initialize_data_quality_reference_cache() -> pd.DataFrame:
    """
    Initialize reference dataframe and category caches for runtime checks.
    """
    cfg = get_data_quality_settings()
    ref_df = load_reference_frame()
    set_reference_frame_cache(ref_df)

    category_cache = build_reference_category_cache(
        ref_df,
        categorical_reference_features=cfg.get(
            "categorical_reference_features", []
        ),
    )
    set_reference_category_cache(category_cache)

    logger.info(
        "Data quality reference cache initialized | "
        f"rows={len(ref_df)} | "
        f"cached_features={list(category_cache.keys())}"
    )
    return ref_df


def summarize_missingness(df: pd.DataFrame) -> dict:
    """
    Compute per-column missing value rates.
    """
    metrics: dict[str, float] = {}

    for col in df.columns:
        metrics[f"missing_rate__{col}"] = float(df[col].isna().mean())

    return metrics


def summarize_unseen_categories(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    categorical_reference_features: list[str],
) -> dict:
    """
    Compare categorical values against the reference dataframe.
    """
    metrics: dict[str, int | str] = {}

    if ref_df.empty:
        return metrics

    for feature in categorical_reference_features:
        if feature not in df.columns:
            logger.warning(f"Categorical feature missing in request data: {feature}")
            continue

        if feature not in ref_df.columns:
            logger.warning(
                f"Categorical reference feature missing in reference data: {feature}"
            )
            continue

        ref_values = set(ref_df[feature].dropna().astype(str).unique())
        cur_values = set(df[feature].dropna().astype(str).unique())

        unseen = sorted(cur_values - ref_values)

        metrics[f"unseen_category_count__{feature}"] = int(len(unseen))
        metrics[f"unseen_categories__{feature}"] = ",".join(unseen[:20])

    return metrics


def summarize_unseen_categories_cached(
    df: pd.DataFrame,
    reference_categories: dict[str, set[str]],
    categorical_reference_features: list[str],
) -> dict:
    """
    Compare categorical values against a cached category reference map.
    """
    metrics: dict[str, int | str] = {}

    if not reference_categories:
        return metrics

    for feature in categorical_reference_features:
        if feature not in df.columns:
            logger.warning(f"Categorical feature missing in request data: {feature}")
            continue

        ref_values = reference_categories.get(feature)
        if ref_values is None:
            logger.warning(
                f"Categorical reference feature missing in cache: {feature}"
            )
            continue

        cur_values = set(df[feature].dropna().astype(str).unique())
        unseen = sorted(cur_values - ref_values)

        metrics[f"unseen_category_count__{feature}"] = int(len(unseen))
        metrics[f"unseen_categories__{feature}"] = ",".join(unseen[:20])

    return metrics


def determine_quality_status(metrics: dict) -> str:
    """
    Determine the overall quality status from computed quality metrics.
    """
    for key, value in metrics.items():
        if key.startswith("missing_rate__") and isinstance(value, float) and value > 0:
            return "warning"

        if (
            key.startswith("unseen_category_count__")
            and isinstance(value, int)
            and value > 0
        ):
            return "warning"

    return "ok"


def summarize_data_quality(df: pd.DataFrame) -> dict:
    """
    Summarize data quality for a dataframe using the persisted reference data.
    """
    cfg = get_data_quality_settings()

    if not cfg.get("enabled", True):
        return {
            "row_count": int(len(df)),
            "quality_status": "disabled",
        }

    summary: dict[str, object] = {
        "row_count": int(len(df)),
    }

    if df.empty:
        summary["quality_status"] = "empty"
        return summary

    ref_df = load_reference_frame()

    summary.update(summarize_missingness(df))
    summary.update(
        summarize_unseen_categories(
            df,
            ref_df,
            categorical_reference_features=cfg.get(
                "categorical_reference_features", []
            ),
        )
    )

    summary["quality_status"] = determine_quality_status(summary)
    return summary


def summarize_data_quality_runtime(
    df: pd.DataFrame,
    reference_categories: dict[str, set[str]] | None = None,
) -> dict:
    """
    Summarize data quality for runtime inference requests.

    Runtime checks use a cached category reference map to avoid repeated reads.
    """
    cfg = get_data_quality_settings()

    if not cfg.get("enabled", True):
        return {
            "row_count": int(len(df)),
            "quality_status": "disabled",
        }

    summary: dict[str, object] = {
        "row_count": int(len(df)),
    }

    if df.empty:
        summary["quality_status"] = "empty"
        return summary

    categories = (
        reference_categories
        if reference_categories is not None
        else _REFERENCE_CATEGORY_CACHE
    )

    summary.update(summarize_missingness(df))
    summary.update(
        summarize_unseen_categories_cached(
            df,
            categories,
            categorical_reference_features=cfg.get(
                "categorical_reference_features", []
            ),
        )
    )

    summary["quality_status"] = determine_quality_status(summary)
    return summary


def append_data_quality_history(summary: dict) -> pd.DataFrame:
    """
    Append a data quality summary to the quality history table.

    The output path can be local or GCS-backed.
    """
    output_path = _history_path()

    row = pd.DataFrame([summary])
    row["timestamp"] = datetime.now(timezone.utc)

    if file_exists(output_path):
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        ensure_dir(MONITORING_PATH)
        combined = row

    combined.to_parquet(output_path, index=False)
    return row


def log_data_quality_runtime(
    df: pd.DataFrame,
    reference_categories: dict[str, set[str]] | None = None,
) -> dict:
    """
    Run and log a runtime data quality check without persisting history.
    """
    summary = summarize_data_quality_runtime(
        df=df,
        reference_categories=reference_categories,
    )

    logger.info(
        "Runtime data quality check finished | "
        f"quality_status={summary.get('quality_status')} | "
        f"row_count={summary.get('row_count')}"
    )

    return summary


def log_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a data quality check and optionally persist the result.
    """
    cfg = get_data_quality_settings()
    summary = summarize_data_quality(df)

    if cfg.get("persist_history", False):
        result_df = append_data_quality_history(summary)
    else:
        result_df = pd.DataFrame([summary])

    logger.info(
        "Data quality check finished | "
        f"quality_status={summary.get('quality_status')} | "
        f"row_count={summary.get('row_count')}"
    )

    return result_df