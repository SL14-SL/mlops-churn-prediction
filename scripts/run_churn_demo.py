from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from src.configs.loader import get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path(get_path("raw_data"))
SIMULATION_FILE = RAW_DATA_PATH / "simulation_ground_truth.csv"
MONITORING_PATH = Path(get_path("monitoring"))
GROUND_TRUTH_BATCH_DIR = MONITORING_PATH / "ground_truth_batches"

def remaining_rows() -> int:
    if not SIMULATION_FILE.exists():
        return 0
    return len(pd.read_csv(SIMULATION_FILE))

def has_released_labels() -> bool:
    return (
        GROUND_TRUTH_BATCH_DIR.exists()
        and any(GROUND_TRUTH_BATCH_DIR.glob("ground_truth_churn*.csv"))
    )


def run_command(cmd: list[str], description: str) -> None:
    logger.info("🚀 %s", description)

    subprocess.run(
        ["uv", "run", "--no-sync", *cmd],
        check=True,
    )


def run_demo(
    *,
    batch_size: int,
    max_days: int,
    label_delay_days: int,
) -> None:
    for day in range(1, max_days + 1):
        remaining = remaining_rows()

        if remaining <= 0:
            logger.info("🏁 Simulation pool is empty. Demo finished.")
            break

        logger.info("=" * 80)
        logger.info("📅 Simulation Day %s | remaining_rows=%s", day, remaining)
        logger.info("=" * 80)

        # 1. Score new customers and store them as pending labels.
        run_command(
            [
                "python",
                "scripts/simulate_churn_batch.py",
                "--batch-size",
                str(batch_size),
                "--simulation-day",
                str(day),
                "--label-delay-days",
                str(label_delay_days),
            ],
            f"Scoring churn batch for day {day}",
        )

        # 2. Release labels that became available today.
        run_command(
            [
                "python",
                "scripts/release_churn_labels.py",
                "--simulation-day",
                str(day),
            ],
            f"Releasing delayed labels for day {day}",
        )

        # 3. Evaluate only if labels are available.
        if not has_released_labels():
            logger.info(
                "No released labels available yet. Skipping performance evaluation and retraining."
            )
            day += 1
            continue

        run_command(
            ["python", "scripts/run_performance_demo.py"],
            f"Evaluating churn performance for day {day}",
        )

        run_command(
            ["python", "flows/auto_retrain_flow.py"],
            f"Running auto-retrain decision for day {day}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-days", type=int, default=10)
    parser.add_argument("--label-delay-days", type=int, default=1)

    args = parser.parse_args()

    run_demo(
        batch_size=args.batch_size,
        max_days=args.max_days,
        label_delay_days=args.label_delay_days,
    )