from __future__ import annotations

import pandas as pd

from mlops_churn_prediction.configs.loader import get_path
from mlops_churn_prediction.storage.filesystem import ensure_dir
from scripts.business_policy_analysis import load_analysis_data, simulate_policy


def main() -> None:
    df = load_analysis_data()
    thresholds = [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30]
    curve = pd.DataFrame(
        [simulate_policy(df, min_expected_profit=threshold) for threshold in thresholds]
    )

    monitoring_path = get_path("monitoring")
    ensure_dir(monitoring_path)
    output = f"{monitoring_path}/profit_curve.parquet"
    curve.to_parquet(output, index=False)

    print(curve.to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
