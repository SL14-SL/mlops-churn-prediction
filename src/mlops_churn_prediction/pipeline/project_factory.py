from collections.abc import Mapping
from typing import Any

from mlops_churn_prediction.inference.releases.churn_input_provider import (
    ChurnServingReleaseInputProvider,
)

from .adapters import (
    ChurnDataIngestor,
    ChurnDatasetSplitter,
    ChurnFeatureBuilder,
    ChurnModelArtifactLogger,
    ChurnModelEvaluator,
    ChurnModelTrainer,
)
from .factory import build_training_pipeline
from .service import TrainingPipeline


def build_project_training_pipeline(
    config: Mapping[str, Any],
) -> TrainingPipeline:
    """Build the complete churn training pipeline."""
    return build_training_pipeline(
        ingestor=ChurnDataIngestor(),
        feature_builder=ChurnFeatureBuilder(),
        splitter=ChurnDatasetSplitter(),
        trainer=ChurnModelTrainer(),
        evaluator=ChurnModelEvaluator(),
        model_logger=(
            ChurnModelArtifactLogger()
        ),
        release_input_provider=(
            ChurnServingReleaseInputProvider()
        ),
        config=config,
    )