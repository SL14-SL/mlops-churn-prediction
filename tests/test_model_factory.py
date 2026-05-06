from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pytest

from src.training.model_factory import build_model


def test_build_gradient_boosting_model():
    cfg = {
        "type": "gradient_boosting",
        "params": {"n_estimators": 10},
    }

    model = build_model(cfg)

    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 10


def test_build_random_forest_model():
    cfg = {
        "type": "random_forest",
        "params": {"n_estimators": 10},
    }

    model = build_model(cfg)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10


def test_build_unsupported_model_raises():
    cfg = {
        "type": "linear_regression",
        "params": {},
    }

    with pytest.raises(ValueError, match="Unsupported model type"):
        build_model(cfg)