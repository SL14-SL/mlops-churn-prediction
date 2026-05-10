from __future__ import annotations

import fsspec

from src.configs.loader import ensure_dir, get_path


def _remove_path(path: str) -> None:
    """
    Remove a local or GCS path recursively.
    """
    fs, fs_path = fsspec.core.url_to_fs(path)

    if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)


def cleanup():
    """
    Cleans up all local data and MLflow artifacts for a fresh start.

    Includes:
    - feature/split/validated datasets
    - simulation pool
    - generated training batches
    - local MLflow artifacts
    """
    # 1. Clear Data Directories
    folders_to_clear = ["splits", "features", "validated_data"]

    for folder in folders_to_clear:
        path = get_path(folder)

        fs, fs_path = fsspec.core.url_to_fs(path)

        if fs.exists(fs_path):
            print(f"🧹 Clearing {path}...")
            fs.rm(fs_path, recursive=True)

        ensure_dir(path)

    # 2. Clear New Batches and the Simulation Pool
    raw_path = get_path("raw_data")

    # Delete the simulation pool file (the "future")
    sim_pool_file = f"{raw_path}/simulation_ground_truth.csv"

    fs, sim_fs_path = fsspec.core.url_to_fs(sim_pool_file)

    if fs.exists(sim_fs_path):
        print(f"🧹 Removing simulation pool: {sim_pool_file}")
        fs.rm(sim_fs_path)

    # Delete existing batches
    new_batches = f"{raw_path}/new_batches"

    batch_fs, batch_fs_path = fsspec.core.url_to_fs(new_batches)

    if batch_fs.exists(batch_fs_path):
        print(f"🧹 Clearing new batches in {new_batches}...")
        batch_fs.rm(batch_fs_path, recursive=True)

    ensure_dir(new_batches)

    # 3. Remove local MLflow Database and Artifacts
    # Intentionally local-only.
    local_mlflow_db = "mlflow.db"
    local_mlruns = "mlruns"

    local_fs = fsspec.filesystem("file")

    if local_fs.exists(local_mlflow_db):
        print("🧹 Removing mlflow.db...")
        local_fs.rm(local_mlflow_db)

    if local_fs.exists(local_mlruns):
        print("🧹 Removing mlruns directory...")
        local_fs.rm(local_mlruns, recursive=True)

    print("✨ System reset complete. Run 'training_flow --force' next.")


if __name__ == "__main__":
    cleanup()