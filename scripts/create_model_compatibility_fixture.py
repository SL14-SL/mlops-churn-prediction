"""Create a frozen synthetic model fixture once, outside CI."""

from importlib.metadata import version
import json
from pathlib import Path

import mlflow.pyfunc
from mlflow.models import infer_signature
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "model_compatibility"

BASELINE = {
    "scikit-learn": "1.8.0",
    "numpy": "2.3.5",
    "scipy": "1.17.0",
}


class ReferenceModel(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, context, model_input, params=None):
        return self.estimator.predict_proba(model_input)[:, 1]


def main():
    if FIXTURE.exists():
        raise RuntimeError(
            "Fixture already exists. Do not regenerate it during updates."
        )

    for package, expected in BASELINE.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(
                f"Generate the baseline with {package}=={expected}; "
                f"installed: {actual}"
            )

    features, labels = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    frame = pd.DataFrame(
        features,
        columns=[f"feature_{i}" for i in range(6)],
    )

    estimator = GradientBoostingClassifier(
        n_estimators=12,
        max_depth=2,
        random_state=42,
    )
    estimator.fit(frame.iloc[:80], labels[:80])

    probe = frame.iloc[80:].reset_index(drop=True)
    probabilities = estimator.predict_proba(probe)[:, 1]

    packages = [
        "mlflow",
        "cloudpickle",
        "pandas",
        "scikit-learn",
        "numpy",
        "scipy",
    ]
    training_versions = {
        package: version(package)
        for package in packages
    }

    FIXTURE.mkdir(parents=True)
    mlflow.pyfunc.save_model(
        path=str(FIXTURE / "model"),
        python_model=ReferenceModel(estimator),
        signature=infer_signature(probe, probabilities),
        input_example=probe.head(2),
        pip_requirements=[
            f"{package}=={package_version}"
            for package, package_version in training_versions.items()
        ],
    )

    reference = {
        "training_versions": training_versions,
        "columns": probe.columns.tolist(),
        "inputs": probe.to_numpy().tolist(),
        "expected_probabilities": probabilities.tolist(),
    }
    (FIXTURE / "reference.json").write_text(
        json.dumps(reference, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created frozen model fixture: {FIXTURE}")


if __name__ == "__main__":
    main()