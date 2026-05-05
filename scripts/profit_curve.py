from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.loader import get_path
from src.monitoring.config import get_business_settings


PREDICTIONS_PATH = Path(get_path("predictions"))
MONITORING_PATH = Path(get_path("monitoring"))

INFERENCE_LOG_FILE = PREDICTIONS_PATH / "inference_log.parquet"
CUMULATIVE_GT_FILE = MONITORING_PATH / "cumulative_ground_truth.csv"
OUTPUT_FILE = MONITORING_PATH / "profit_curve.parquet"


def normalize_customer_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" in df.columns and "customerid" not in df.columns:
        df = df.rename(columns={"customerID": "customerid"})

    return df


def load_labeled_predictions() -> pd.DataFrame:
    if not INFERENCE_LOG_FILE.exists():
        raise FileNotFoundError(f"Prediction log not found: {INFERENCE_LOG_FILE}")

    if not CUMULATIVE_GT_FILE.exists():
        raise FileNotFoundError(f"Ground truth file not found: {CUMULATIVE_GT_FILE}")

    predictions = pd.read_parquet(INFERENCE_LOG_FILE)
    ground_truth = pd.read_csv(CUMULATIVE_GT_FILE)

    predictions = normalize_customer_id_column(predictions)
    ground_truth = normalize_customer_id_column(ground_truth)

    if "prediction_id" not in predictions.columns:
        raise KeyError("Prediction log must contain `prediction_id`.")

    if "prediction_id" not in ground_truth.columns:
        raise KeyError("Ground truth must contain `prediction_id`.")

    if "churn" not in ground_truth.columns:
        raise KeyError("Ground truth must contain `churn`.")

    predictions["churn_probability"] = pd.to_numeric(
        predictions.get("churn_probability", predictions.get("prediction")),
        errors="coerce",
    )

    ground_truth["churn"] = pd.to_numeric(ground_truth["churn"], errors="coerce")

    df = predictions.merge(
        ground_truth[["prediction_id", "churn"]],
        on="prediction_id",
        how="inner",
    )

    df = df.dropna(subset=["churn_probability", "churn"]).copy()
    df["churn"] = df["churn"].astype(int)

    if df.empty:
        raise ValueError("No labeled predictions available.")

    return df


def choose_action(
    p: float,
    *,
    customer_value: float,
    cost_contact: float,
    cost_discount: float,
    contact_uplift: float,
    discount_uplift: float,
    min_expected_profit: float,
) -> tuple[str, float, float, float]:
    send_email_ev = p * customer_value * contact_uplift - cost_contact
    offer_discount_ev = p * customer_value * discount_uplift - cost_discount

    action_values = {
        "send_email": send_email_ev,
        "offer_discount": offer_discount_ev,
        "no_action": 0.0,
    }

    best_action = max(action_values, key=action_values.get)
    best_ev = float(action_values[best_action])

    if best_action == "no_action" or best_ev < min_expected_profit:
        return "no_action", 0.0, 0.0, 0.0

    if best_action == "send_email":
        return best_action, best_ev, cost_contact, contact_uplift

    return best_action, best_ev, cost_discount, discount_uplift


def simulate_profit_curve(
    df: pd.DataFrame,
    min_expected_profit_values: list[float],
) -> pd.DataFrame:
    business_cfg = get_business_settings()

    rows = []

    for min_expected_profit in min_expected_profit_values:
        simulated = df.copy()

        decisions = simulated["churn_probability"].apply(
            lambda p: choose_action(
                float(p),
                customer_value=business_cfg["customer_value"],
                cost_contact=business_cfg["cost_contact"],
                cost_discount=business_cfg["cost_discount"],
                contact_uplift=business_cfg["contact_uplift"],
                discount_uplift=business_cfg["discount_uplift"],
                min_expected_profit=min_expected_profit,
            )
        )

        simulated[
            [
                "sim_action",
                "sim_expected_value",
                "sim_cost",
                "sim_uplift",
            ]
        ] = pd.DataFrame(decisions.tolist(), index=simulated.index)

        actioned = simulated["sim_action"].ne("no_action")

        simulated["sim_expected_saved_value"] = (
            simulated["churn_probability"]
            * business_cfg["customer_value"]
            * simulated["sim_uplift"]
        )

        simulated["sim_expected_profit"] = (
            simulated["sim_expected_saved_value"] - simulated["sim_cost"]
        )

        simulated.loc[~actioned, "sim_expected_profit"] = 0.0
        simulated.loc[~actioned, "sim_expected_saved_value"] = 0.0

        simulated["sim_realized_saved_value"] = (
            simulated["churn"]
            * business_cfg["customer_value"]
            * simulated["sim_uplift"]
        )

        simulated["sim_realized_profit"] = (
            simulated["sim_realized_saved_value"] - simulated["sim_cost"]
        )

        simulated.loc[~actioned, "sim_realized_profit"] = 0.0
        simulated.loc[~actioned, "sim_realized_saved_value"] = 0.0

        rows.append(
            {
                "min_expected_profit": float(min_expected_profit),
                "n_samples": int(len(simulated)),
                "actions_count": int(actioned.sum()),
                "action_rate": float(actioned.mean()),
                "send_email_count": int(simulated["sim_action"].eq("send_email").sum()),
                "offer_discount_count": int(
                    simulated["sim_action"].eq("offer_discount").sum()
                ),
                "expected_profit": float(simulated["sim_expected_profit"].sum()),
                "realized_profit": float(simulated["sim_realized_profit"].sum()),
                "expected_profit_per_action": float(
                    simulated.loc[actioned, "sim_expected_profit"].mean()
                )
                if actioned.any()
                else 0.0,
                "realized_profit_per_action": float(
                    simulated.loc[actioned, "sim_realized_profit"].mean()
                )
                if actioned.any()
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = load_labeled_predictions()

    min_expected_profit_values = [
        0,
        1,
        2,
        3,
        5,
        7.5,
        10,
        15,
        20,
        25,
        30,
    ]

    curve = simulate_profit_curve(df, min_expected_profit_values)

    MONITORING_PATH.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(OUTPUT_FILE, index=False)

    print("PROFIT_CURVE=")
    print(curve.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()