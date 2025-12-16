from ..interfaces.base_preprocessor import BasePreprocessor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LogMinMaxScaler(BasePreprocessor):
    """
    Applies feature-wise normalization for numeric features.

    Strategy:
    ----------
    - Binary features (<= `binary_threshold` unique values) are left unchanged.
    - Continuous features are scaled to [0, 1] using MinMaxScaler.
    - Log-transform is applied ONLY to features that are strictly positive.
      This avoids invalid log operations for features such as audio loudness
      (which may contain negative values and is already on a logarithmic scale).

    This design is especially suitable for LightFM and similar models, where:
    - Feature magnitudes should be comparable
    - Strict normality is not required
    """

    def __init__(
        self,
        feature_cols=None,
        id_cols=("userID", "artistID", "unified_artist_id"),
        binary_threshold=2,
        log_offset=1e-10,
    ):
        """
        Args:
            feature_cols:
                Explicit list of features to process. If None, numeric columns
                are auto-detected (excluding ID columns).
            id_cols:
                Columns treated as identifiers and excluded from processing.
            binary_threshold:
                Maximum number of unique values for a feature to be considered binary.
            log_offset:
                Small constant added before log transform to avoid log(0).
                Used only for strictly positive features.
        """
        self.feature_cols = feature_cols
        self.id_cols = id_cols
        self.binary_threshold = binary_threshold
        self.log_offset = log_offset

        # Fitted attributes
        self.scalers_ = {}  # column -> fitted MinMaxScaler
        self.continuous_cols_ = []  # all continuous (non-binary) features
        self.binary_cols_ = []  # binary features
        self.log_cols_ = []  # continuous features using log + MinMax
        self.no_log_cols_ = []  # continuous features using MinMax only

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _identify_feature_types(self, X: pd.DataFrame):
        """
        Identify candidate numeric features and split them into
        continuous vs binary based on cardinality.
        """
        if self.feature_cols is not None:
            candidate_cols = [c for c in self.feature_cols if c in X.columns]
        else:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            candidate_cols = [c for c in numeric_cols if c not in self.id_cols]

        continuous_cols = []
        binary_cols = []

        for col in candidate_cols:
            if X[col].nunique() <= self.binary_threshold:
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        return continuous_cols, binary_cols

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit feature-wise MinMax scalers.

        - Continuous strictly-positive features:
            log(x + offset) -> MinMax
        - Continuous features with zero or negative values:
            MinMax only
        """
        X = X.copy()

        # Identify feature types
        self.continuous_cols_, self.binary_cols_ = self._identify_feature_types(X)

        if not self.continuous_cols_ and not self.binary_cols_:
            raise ValueError(
                "No valid features found for scaling. "
                f"ID cols: {self.id_cols}, Available cols: {X.columns.tolist()}"
            )

        # Split continuous features by log applicability
        for col in self.continuous_cols_:
            if (X[col] <= 0).any():
                # Log transform would be invalid or meaningless
                # (e.g. loudness in dB)
                self.no_log_cols_.append(col)
            else:
                self.log_cols_.append(col)

        # Fit scalers for log-transformed features
        for col in self.log_cols_:
            values = np.log(X[col] + self.log_offset)

            scaler = MinMaxScaler()
            scaler.fit(values.values.reshape(-1, 1))

            self.scalers_[col] = scaler

        # Fit scalers for non-log features
        for col in self.no_log_cols_:
            scaler = MinMaxScaler()
            scaler.fit(X[col].values.reshape(-1, 1))

            self.scalers_[col] = scaler

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted transformations to input data.

        Returns:
            DataFrame with scaled continuous features and untouched binary features.
        """
        if not self.scalers_:
            raise RuntimeError("Must call fit() before transform()")

        X = X.copy()

        # Log + MinMax features
        for col in self.log_cols_:
            if col not in X.columns:
                raise KeyError(f"Column '{col}' missing during transform")

            values = np.log(X[col] + self.log_offset)
            X[col] = (
                self.scalers_[col].transform(values.values.reshape(-1, 1)).flatten()
            )

        # MinMax-only features (e.g. loudness)
        for col in self.no_log_cols_:
            if col not in X.columns:
                raise KeyError(f"Column '{col}' missing during transform")

            X[col] = (
                self.scalers_[col].transform(X[col].values.reshape(-1, 1)).flatten()
            )

        # Binary features are left unchanged
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_feature_info(self):
        """
        Return information about how features were processed.
        Useful for debugging and experiment tracking.
        """
        if not self.scalers_:
            raise RuntimeError("Must call fit() before get_feature_info()")

        return {
            "log_scaled_features": self.log_cols_,
            "minmax_only_features": self.no_log_cols_,
            "binary_features": self.binary_cols_,
            "n_log_scaled": len(self.log_cols_),
            "n_minmax_only": len(self.no_log_cols_),
            "n_binary": len(self.binary_cols_),
        }
