from __future__ import annotations

import pandas as pd

from mlops_churn_prediction.configs.loader import get_path
from mlops_churn_prediction.monitoring.config import get_business_settings
from mlops_churn_prediction.storage.filesystem import ensure_dir
from scripts.business_policy_analysis import load_analysis_data, simulate_policy


def main() -> None:
    df = load_analysis_data()
    threshold = float(get_business_settings()["min_expected_profit"])
    rows = []

    for discount in [0.1, 0.2, 0.3, 0.4]:
        for contact in [0.05, 0.1, 0.15]:
            result = simulate_policy(
                df,
                min_expected_profit=threshold,
                discount_uplift=discount,
                contact_uplift=contact,
            )
            # Preserve the column expected by existing dashboard code.
            result["actions"] = result["actions_count"]
            rows.append(result)

    results = pd.DataFrame(rows)
    monitoring_path = get_path("monitoring")
    ensure_dir(monitoring_path)
    output = f"{monitoring_path}/uplift_sensitivity.csv"
    results.to_csv(output, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
