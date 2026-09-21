from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_churn_prediction.data.contracts import (
    DatasetCollection,
    DatasetSplits,
)
from mlops_churn_prediction.data.features.pipeline import (
    build_feature_table,
)
from mlops_churn_prediction.data.raw.ingest import (
    ingest as ingest_churn_data,
)
from mlops_churn_prediction.data.splits.split import (
    split_features,
)


class ChurnDataIngestor:
    """Adapt the existing churn ingestion lifecycle."""

    def ingest(
        self,
        config: Mapping[str, Any],
    ) -> DatasetCollection:
        """
        Ingest churn data and return its validated dataset.

        The legacy ingestion entrypoint currently resolves its
        environment-specific configuration internally.
        """
        del config
        return ingest_churn_data()


class ChurnFeatureBuilder:
    """Adapt churn feature engineering to the shared contract."""

    def build_features(
        self,
        datasets: DatasetCollection,
        config: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Build the model-ready churn feature table."""
        return build_feature_table(
            datasets,
            config,
        )


class ChurnDatasetSplitter:
    """Adapt churn dataset splitting to the shared contract."""

    def split(
        self,
        features: pd.DataFrame,
        config: Mapping[str, Any],
    ) -> DatasetSplits:
        """Create stratified churn train and validation splits."""
        return split_features(
            features,
            config,
        )