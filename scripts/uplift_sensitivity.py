from __future__ import annotations

import pandas as pd

from src.configs.loader import ensure_dir, get_path

PREDICTIONS_PATH = get_path("predictions")
MONITORING_PATH = get_path("monitoring")

INFERENCE_LOG_FILE = f"{PREDICTIONS_PATH}/inference_log.parquet"
CUMULATIVE_GT_FILE = f"{MONITORING_PATH}/cumulative_ground_truth.csv"


def load_labeled_predictions() -> pd.DataFrame:
    """
    Load prediction logs and cumulative ground truth labels.

    The paths may point either to the local filesystem or to a remote filesystem
    such as GCS. Therefore, paths are kept as strings instead of pathlib.Path
    objects.
    """
    preds = pd.read_parquet(INFERENCE_LOG_FILE)
    gt = pd.read_csv(CUMULATIVE_GT_FILE)

    preds["churn_probability"] = pd.to_numeric(
        preds.get("churn_probability", preds.get("prediction")),
        errors="coerce",
    )

    gt["churn"] = pd.to_numeric(gt["churn"], errors="coerce")

    df = preds.merge(
        gt[["prediction_id", "churn"]],
        on="prediction_id",
        how="inner",
    )

    df = df.dropna(subset=["churn_probability", "churn"]).copy()
    df["churn"] = df["churn"].astype(int)

    return df


def simulate(
    df: pd.DataFrame,
    *,
    discount_uplift: float,
    contact_uplift: float,
    min_expected_profit: float = 0,
    customer_value: float = 100,
    cost_contact: float = 2,
    cost_discount: float = 10,
) -> dict:
    """
    Simulate business outcomes for a given set of uplift assumptions.

    The decision logic is intentionally unchanged: for every prediction, the
    action with the highest expected value is selected unless it falls below the
    minimum expected profit threshold.
    """
    sim = df.copy()

    def decide(p):
        email_ev = p * customer_value * contact_uplift - cost_contact
        discount_ev = p * customer_value * discount_uplift - cost_discount

        values = {
            "send_email": email_ev,
            "offer_discount": discount_ev,
            "no_action": 0,
        }

        best = max(values, key=values.get)
        best_val = values[best]

        if best == "no_action" or best_val < min_expected_profit:
            return "no_action", 0, 0, 0

        if best == "send_email":
            return best, best_val, cost_contact, contact_uplift

        return best, best_val, cost_discount, discount_uplift

    res = sim["churn_probability"].apply(decide)

    sim[["action", "ev", "cost", "uplift"]] = pd.DataFrame(
        res.tolist(),
        index=sim.index,
    )

    actioned = sim["action"] != "no_action"

    sim["realized_profit"] = (
        sim["churn"] * customer_value * sim["uplift"] - sim["cost"]
    )
    sim.loc[~actioned, "realized_profit"] = 0

    return {
        "discount_uplift": discount_uplift,
        "contact_uplift": contact_uplift,
        "actions": int(actioned.sum()),
        "realized_profit": float(sim["realized_profit"].sum()),
    }


def main() -> None:
    """
    Run the uplift sensitivity analysis and persist the result table.

    The output directory is created through the project helper so it works for
    both local paths and GCS-backed paths.
    """
    df = load_labeled_predictions()

    results = []

    for discount in [0.1, 0.2, 0.3, 0.4]:
        for contact in [0.05, 0.1, 0.15]:
            results.append(
                simulate(
                    df,
                    discount_uplift=discount,
                    contact_uplift=contact,
                )
            )

    res_df = pd.DataFrame(results)
    print(res_df)

    ensure_dir(MONITORING_PATH)

    out = f"{MONITORING_PATH}/uplift_sensitivity.csv"
    res_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()