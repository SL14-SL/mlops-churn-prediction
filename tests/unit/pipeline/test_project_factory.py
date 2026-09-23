from unittest.mock import MagicMock

import pytest

from mlops_churn_prediction.inference.releases.churn_input_provider import (
    ChurnServingReleaseInputProvider,
)
from mlops_churn_prediction.pipeline import (
    project_factory,
)
from mlops_churn_prediction.pipeline.adapters import (
    ChurnDataIngestor,
    ChurnDatasetSplitter,
    ChurnFeatureBuilder,
    ChurnModelArtifactLogger,
    ChurnModelEvaluator,
    ChurnModelTrainer,
)


def test_builds_complete_churn_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_pipeline = MagicMock()
    build_pipeline = MagicMock(
        return_value=expected_pipeline
    )

    monkeypatch.setattr(
        project_factory,
        "build_training_pipeline",
        build_pipeline,
    )

    config = {
        "paths": {
            "pipeline_runs": (
                "data/pipeline-runs"
            ),
        },
    }

    result = (
        project_factory
        .build_project_training_pipeline(
            config
        )
    )

    assert result is expected_pipeline

    call = build_pipeline.call_args
    arguments = call.kwargs

    assert isinstance(
        arguments["ingestor"],
        ChurnDataIngestor,
    )
    assert isinstance(
        arguments["feature_builder"],
        ChurnFeatureBuilder,
    )
    assert isinstance(
        arguments["splitter"],
        ChurnDatasetSplitter,
    )
    assert isinstance(
        arguments["trainer"],
        ChurnModelTrainer,
    )
    assert isinstance(
        arguments["evaluator"],
        ChurnModelEvaluator,
    )
    assert isinstance(
        arguments["model_logger"],
        ChurnModelArtifactLogger,
    )
    assert isinstance(
        arguments["release_input_provider"],
        ChurnServingReleaseInputProvider,
    )
    assert arguments["config"] is config


def test_builds_real_pipeline_repository(
    tmp_path,
) -> None:
    pipeline_runs = (
        tmp_path / "pipeline-runs"
    )

    pipeline = (
        project_factory
        .build_project_training_pipeline(
            {
                "paths": {
                    "pipeline_runs": str(
                        pipeline_runs
                    ),
                },
            }
        )
    )

    assert (
        pipeline.run_repository.root_path
        == str(pipeline_runs)
    )