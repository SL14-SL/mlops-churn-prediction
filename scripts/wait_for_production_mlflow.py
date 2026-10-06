import os
import time

import requests

from mlflow_cloud_run_auth import CloudRunAuth


def main() -> None:
    base_url = os.environ["MLFLOW_UI_URL"].rstrip("/")
    auth = CloudRunAuth()

    for attempt in range(1, 61):
        try:
            response = requests.get(
                f"{base_url}/health",
                auth=auth,
                timeout=15,
            )

            if response.status_code == 200:
                print("MLflow is ready; authenticated health check passed.")
                return

            if response.status_code in {401, 403}:
                raise RuntimeError(f"MLflow authentication failed: HTTP {response.status_code}")

            print(f"MLflow not ready: HTTP {response.status_code}; attempt {attempt}")

        except requests.RequestException as exc:
            print(f"MLflow not reachable: {type(exc).__name__}; attempt {attempt}")

        if attempt < 60:
            time.sleep(5)

    raise RuntimeError("MLflow did not become healthy.")


if __name__ == "__main__":
    main()
