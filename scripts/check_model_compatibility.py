"""Check a committed model artifact without training or regenerating it."""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from mlops_churn_prediction.inference.model_loader import load_model_by_type


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "model_compatibility"


def main():
    reference = json.loads(
        (FIXTURE / "reference.json").read_text(encoding="utf-8")
    )
    frame = pd.DataFrame(
        reference["inputs"],
        columns=reference["columns"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = load_model_by_type(
            str(FIXTURE / "model"),
            "gradient_boosting",
        )
        actual = np.asarray(model.predict(frame), dtype=float)

    expected = np.asarray(
        reference["expected_probabilities"],
        dtype=float,
    )

    if actual.shape != expected.shape:
        raise AssertionError(
            f"Prediction shape changed: {actual.shape} != {expected.shape}"
        )
    if not np.isfinite(actual).all():
        raise AssertionError("Predictions contain non-finite values.")

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )
    print(
        "Persisted model compatibility passed: "
        f"{len(actual)} reference predictions."
    )


if __name__ == "__main__":
    main()