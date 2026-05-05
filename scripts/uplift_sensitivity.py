from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.configs.loader import get_path

PREDICTIONS_PATH = Path(get_path("predictions"))
MONITORING_PATH = Path(get_path("monitoring"))

INFERENCE_LOG_FILE = PREDICTIONS_PATH / "inference_log.parquet"
CUMULATIVE_GT_FILE = MONITORING_PATH / "cumulative_ground_truth.csv"


def load_labeled_predictions() -> pd.DataFrame:
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
    sim = df.copy()

    def decide(p):
        email_ev = p * customer_value * contact_uplift - cost_contact
        discount_ev = p * customer_value * discount_uplift - cost_discount

        best = max(
            {"send_email": email_ev, "offer_discount": discount_ev, "no_action": 0},
            key=lambda k: {"send_email": email_ev, "offer_discount": discount_ev, "no_action": 0}[k],
        )

        best_val = {"send_email": email_ev, "offer_discount": discount_ev, "no_action": 0}[best]

        if best == "no_action" or best_val < min_expected_profit:
            return "no_action", 0, 0, 0

        if best == "send_email":
            return best, best_val, cost_contact, contact_uplift

        return best, best_val, cost_discount, discount_uplift

    res = sim["churn_probability"].apply(decide)

    sim[["action", "ev", "cost", "uplift"]] = pd.DataFrame(res.tolist(), index=sim.index)

    actioned = sim["action"] != "no_action"

    sim["realized_profit"] = sim["churn"] * customer_value * sim["uplift"] - sim["cost"]
    sim.loc[~actioned, "realized_profit"] = 0

    return {
        "discount_uplift": discount_uplift,
        "contact_uplift": contact_uplift,
        "actions": int(actioned.sum()),
        "realized_profit": float(sim["realized_profit"].sum()),
    }


def main():
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

    out = MONITORING_PATH / "uplift_sensitivity.csv"
    res_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()