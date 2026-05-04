from __future__ import annotations

from prefect import flow, get_run_logger

from flows.training_flow import training_pipeline
from src.monitoring.trigger import should_retrain


@flow(name="Auto Retrain Decision Flow")
def auto_retrain_flow() -> str:
    """Run retraining only when churn monitoring thresholds are violated."""
    logger = get_run_logger()

    if not should_retrain():
        logger.info("No churn performance degradation detected. Skipping retraining.")
        return "skipped"

    logger.warning("Churn performance degradation detected. Triggering training pipeline.")
    result = training_pipeline(force_run=True)

    champion_promoted = bool(result.get("champion_promoted", False)) if result else False

    if champion_promoted:
        logger.info("Retraining finished. New champion model was promoted.")
        return "retrained_and_promoted"

    logger.info("Retraining finished. No new champion was promoted.")
    return "retrained_no_promotion"


if __name__ == "__main__":
    auto_retrain_flow()