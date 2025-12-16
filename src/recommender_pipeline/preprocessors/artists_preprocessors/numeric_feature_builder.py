from ...interfaces.base_preprocessor import BasePreprocessor
import pandas as pd
import numpy as np


class ArtistNumericFeatureBuilder(BasePreprocessor):
    """
    Aggregates track-level data to artist-level by:
    1. Computing mean of numeric features per artist (on non-null values)
    2. Filling remaining nulls with global mean
    3. Taking mode (most common value) of binary features per artist
    4. Returns artist-level data with artistID as index (no duplicates)
    """

    def __init__(
        self,
        artist_col="artistID",
        numeric_cols=None,
        binary_cols_prefix=("genre_", "mood_", "geo_"),
    ):
        """
        Args:
            artist_col: Column identifying the artist
            numeric_cols: Explicit list of numeric columns to aggregate.
                         If None, auto-detects all numeric columns.
            binary_cols_prefix: Prefixes for binary columns to aggregate with mode
        """
        self.artist_col = artist_col
        self.numeric_cols = numeric_cols
        self.binary_cols_prefix = binary_cols_prefix

    def _get_binary_columns(self, X: pd.DataFrame):
        """Get all columns matching binary prefixes."""
        return [
            c
            for c in X.columns
            if isinstance(c, str)
            and any(c.startswith(p) for p in self.binary_cols_prefix)
        ]

    def _get_numeric_columns(self, X: pd.DataFrame):
        """Get numeric columns to aggregate."""
        binary_cols = self._get_binary_columns(X)

        if self.numeric_cols is not None:
            # Use explicit list
            valid_cols = [
                c
                for c in self.numeric_cols
                if c in X.columns
                and c != self.artist_col
                and c not in binary_cols
                and pd.api.types.is_numeric_dtype(X[c])
            ]
            return valid_cols

        # Auto-detect numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Remove artist ID and binary columns
        numeric_cols = [
            c for c in numeric_cols if c != self.artist_col and c not in binary_cols
        ]

        return numeric_cols

    def _mode_with_tie_break(self, series):
        """
        Return the mode (most common value).
        If tie, return the first mode. If empty, return 0.
        """
        mode_result = series.mode()
        if len(mode_result) == 0:
            return 0  # Default for empty series
        return mode_result.iloc[0]  # Return first mode in case of tie

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform track-level data to artist-level.

        Returns:
            DataFrame with artistID as index, aggregated numeric features (mean),
            and mode of binary features.
        """
        if self.artist_col not in X.columns:
            raise ValueError(
                f"Artist column '{self.artist_col}' not found in DataFrame"
            )

        X = X.copy()

        numeric_cols = self._get_numeric_columns(X)
        binary_cols = self._get_binary_columns(X)

        if not numeric_cols and not binary_cols:
            raise ValueError(
                "No numeric or binary columns found to process. "
                f"Available columns: {X.columns.tolist()}"
            )

        # Build aggregation dictionary
        agg_dict = {}

        # Numeric columns: mean
        for col in numeric_cols:
            agg_dict[col] = "mean"

        # Binary columns: mode
        for col in binary_cols:
            agg_dict[col] = self._mode_with_tie_break

        # Aggregate all at once
        artist_features = X.groupby(self.artist_col).agg(agg_dict)

        # Fill remaining numeric nulls with global mean
        for col in numeric_cols:
            if artist_features[col].isna().any():
                global_mean = X[col].mean()
                artist_features[col] = artist_features[col].fillna(global_mean)

        # Ensure binary columns are integers
        for col in binary_cols:
            artist_features[col] = artist_features[col].fillna(0).astype(int)

        # Ensure index name is set
        artist_features.index.name = self.artist_col

        return artist_features
