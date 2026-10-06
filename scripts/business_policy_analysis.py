from __future__ import annotations

from dataclasses import replace

import pandas as pd

from mlops_churn_prediction.configs.loader import get_path, load_config
from mlops_churn_prediction.inference.decision import DecisionConfig, DecisionEngine


def load_analysis_data() -> pd.DataFrame:
    predictions = pd.read_parquet(f"{get_path('predictions')}/inference_log.parquet")
    ground_truth = pd.read_csv(f"{get_path('monitoring')}/cumulative_ground_truth.csv")

    required = {
        "prediction_id",
        "request_id",
        "customer_value",
        "model_run_id",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    if predictions["prediction_id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs.")

    request_ids = predictions["request_id"].astype("string")
    if request_ids.isna().any() or request_ids.str.strip().eq("").any():
        raise ValueError("Every prediction needs a batch request_id.")

    predictions["request_id"] = request_ids
    probability_column = (
        "churn_probability" if "churn_probability" in predictions.columns else "prediction"
    )
    predictions["churn_probability"] = pd.to_numeric(
        predictions[probability_column], errors="raise"
    )
    predictions["customer_value"] = pd.to_numeric(predictions["customer_value"], errors="raise")

    if predictions[["churn_probability", "customer_value"]].isna().any().any():
        raise ValueError("Missing probability or customer value.")

    if not predictions["churn_probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not predictions["customer_value"].ge(0).all():
        raise ValueError("Customer values must be non-negative.")

    model_counts = predictions.groupby("request_id")["model_run_id"].nunique(dropna=False)
    if model_counts.gt(1).any():
        raise ValueError("A request_id contains predictions from multiple models.")

    ground_truth = ground_truth[["prediction_id", "churn"]].copy()
    ground_truth["churn"] = pd.to_numeric(ground_truth["churn"], errors="raise")
    available = ground_truth["churn"].dropna()
    if not available.isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    return predictions.merge(
        ground_truth,
        on="prediction_id",
        how="left",
        validate="one_to_one",
    )


def simulate_policy(
    df: pd.DataFrame,
    *,
    min_expected_profit: float = 0.0,
    contact_uplift: float | None = None,
    discount_uplift: float | None = None,
) -> dict:
    config = DecisionConfig.from_config(load_config())
    overrides = {}
    if contact_uplift is not None:
        overrides["contact_uplift"] = contact_uplift
    if discount_uplift is not None:
        overrides["discount_uplift"] = discount_uplift
    config = replace(config, **overrides)
    engine = DecisionEngine(config)

    batches = []
    for _, batch in df.groupby("request_id", sort=False):
        batch = batch.copy()
        decisions = engine.decide_batch(
            probs=batch["churn_probability"].tolist(),
            customer_values=batch["customer_value"].tolist(),
        )
        batch["sim_action"] = [d["action"] for d in decisions]
        batch["sim_expected_profit"] = [d["expected_value"] for d in decisions]

        # Apply the analysis threshold after budget allocation.
        # Freed budget is deliberately not reallocated.
        rejected = batch["sim_expected_profit"].le(0) | batch["sim_expected_profit"].lt(
            min_expected_profit
        )
        batch.loc[rejected, "sim_action"] = "no_action"
        batch.loc[rejected, "sim_expected_profit"] = 0.0
        batches.append(batch)

    if not batches:
        raise ValueError("No predictions available.")

    simulated = pd.concat(batches, ignore_index=True)
    labeled = simulated.dropna(subset=["churn"]).copy()
    if labeled.empty:
        raise ValueError("No labeled predictions available.")

    costs = {
        "no_action": 0.0,
        "send_email": config.cost_contact,
        "offer_discount": config.cost_discount,
    }
    uplifts = {
        "no_action": 0.0,
        "send_email": config.contact_uplift,
        "offer_discount": config.discount_uplift,
    }
    labeled["sim_realized_profit"] = labeled["churn"] * labeled["customer_value"] * labeled[
        "sim_action"
    ].map(uplifts) - labeled["sim_action"].map(costs)

    actioned = labeled["sim_action"].ne("no_action")
    actions_count = int(actioned.sum())

    return {
        "min_expected_profit": float(min_expected_profit),
        "contact_uplift": config.contact_uplift,
        "discount_uplift": config.discount_uplift,
        "n_samples": len(labeled),
        "actions_count": actions_count,
        "action_rate": float(actioned.mean()),
        "send_email_count": int(labeled["sim_action"].eq("send_email").sum()),
        "offer_discount_count": int(labeled["sim_action"].eq("offer_discount").sum()),
        "expected_profit": float(labeled["sim_expected_profit"].sum()),
        "realized_profit": float(labeled["sim_realized_profit"].sum()),
        "expected_profit_per_action": (
            float(labeled.loc[actioned, "sim_expected_profit"].mean()) if actions_count else 0.0
        ),
        "realized_profit_per_action": (
            float(labeled.loc[actioned, "sim_realized_profit"].mean()) if actions_count else 0.0
        ),
    }
