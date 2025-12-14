from typing import Iterable, List, Optional

import pandas as pd

from .group_feature_generator_base import GroupFeatureGeneratorBase


class TopValueExtractor(GroupFeatureGeneratorBase):
    """Extracts the top/most frequent value for categorical columns while keeping all rows."""

    def __init__(
        self,
        groupby_col: str = "artist_name",
        categorical_feature_cols: Optional[Iterable[str]] = None,
        feature_prefix: str = "ft_",
    ):
        super().__init__(groupby_col=groupby_col, feature_prefix=feature_prefix)
        self.categorical_feature_cols = list(categorical_feature_cols) if categorical_feature_cols is not None else None
        self._resolved_categorical_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._resolved_categorical_cols = self._resolve_categorical_cols(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        if not self._resolved_categorical_cols:
            self._resolved_categorical_cols = self._resolve_categorical_cols(X)
        categorical_cols = self._resolved_categorical_cols
        if not categorical_cols:
            raise ValueError("No categorical columns available for top value extraction.")
        result_df = X.copy()
        for col in categorical_cols:
            if col in X.columns:
                result_df[f"{self.feature_prefix}{col}_top"] = X.groupby(self.groupby_col)[col].transform(self._mode_safe)
        return result_df

    def _resolve_categorical_cols(self, X: pd.DataFrame) -> List[str]:
        drop_cols = [self.groupby_col] if self.groupby_col in X.columns else []
        return self._resolve_columns(X, self.categorical_feature_cols, drop_cols, numeric=False)

    @staticmethod
    def _mode_safe(series: pd.Series):
        """Safely extract mode (most frequent value) from a series."""
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else None