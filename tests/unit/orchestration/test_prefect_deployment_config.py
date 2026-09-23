from pathlib import Path

import yaml


PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)


def load_deployments() -> list[dict]:
    config_path = (
        PROJECT_ROOT / "prefect.yaml"
    )
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    return config["deployments"]


def test_existing_auto_retrain_deployment_remains() -> None:
    deployments = load_deployments()

    deployment_names = {
        deployment["name"]
        for deployment in deployments
    }

    assert "auto-retrain" in deployment_names


def test_template_training_deployment_is_manual() -> None:
    deployments = load_deployments()

    deployment = next(
        deployment
        for deployment in deployments
        if deployment["name"]
        == "template-training-validation"
    )

    assert deployment["entrypoint"] == (
        "src/mlops_churn_prediction/"
        "orchestration/"
        "template_training_flow.py:"
        "training_flow"
    )
    assert deployment["parameters"] == {
        "run_id": None,
    }
    assert "schedules" not in deployment