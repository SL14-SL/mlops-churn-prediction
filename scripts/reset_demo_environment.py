from __future__ import annotations

import argparse
from pathlib import PurePosixPath

import fsspec

from src.configs.loader import ensure_dir, get_path


RAW_FILES_TO_KEEP = {
    "Telco-Customer-Churn.csv",
}


def _exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def _remove(path: str, *, recursive: bool = True) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)

    if fs.exists(fs_path):
        print(f"🧹 Removing {path}")
        fs.rm(fs_path, recursive=recursive)


def _list_files(path: str) -> list[str]:
    fs, fs_path = fsspec.core.url_to_fs(path)

    if not fs.exists(fs_path):
        return []

    return fs.ls(fs_path, detail=False)


def _to_display_path(path: str, base_path: str) -> str:
    if base_path.startswith("gs://") and not path.startswith("gs://"):
        return f"gs://{path}"
    return path


def clear_directory(path: str, *, recreate: bool = True) -> None:
    """
    Remove a configured local or GCS directory and optionally recreate it.
    """
    _remove(path, recursive=True)

    if recreate:
        ensure_dir(path)


def clear_raw_data_except_source_files(raw_path: str) -> None:
    """
    Clear generated raw-data artifacts while keeping the original source CSV.

    Keeps files listed in RAW_FILES_TO_KEEP at the raw-data root.
    Removes generated subdirectories such as new_batches and quarantine.
    """
    ensure_dir(raw_path)

    for item in _list_files(raw_path):
        item_path = _to_display_path(item, raw_path)
        name = PurePosixPath(item_path).name

        if name in RAW_FILES_TO_KEEP:
            print(f"✅ Keeping raw source file: {item_path}")
            continue

        _remove(item_path, recursive=True)


def reset_demo_environment() -> None:
    """
    Reset the complete demo environment to a clean state.

    This removes generated pipeline artifacts, monitoring histories,
    prediction logs, model reports, simulation batches, and local MLflow files.
    It keeps the original raw dataset CSV.
    """
    raw_path = get_path("raw_data")

    generated_paths = [
        get_path("features"),
        get_path("splits"),
        get_path("validated_data"),
        get_path("predictions"),
        get_path("monitoring"),
        get_path("models"),
        get_path("versioning"),
    ]

    for path in generated_paths:
        clear_directory(path, recreate=True)

    clear_raw_data_except_source_files(raw_path)

    # Local-only MLflow artifacts.
    clear_directory("mlruns", recreate=False)
    _remove("mlflow.db", recursive=False)

    print("✨ Demo environment reset complete.")
    print("Remaining expected artifact: original raw CSV only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reset generated demo artifacts while keeping the raw source CSV."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion without interactive prompt.",
    )

    args = parser.parse_args()

    if not args.yes:
        answer = input(
            "This will delete generated demo artifacts locally or in GCS. Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            raise SystemExit("Reset aborted.")

    reset_demo_environment()