from typing import Iterable, List, Optional

import pandas as pd

from .group_feature_generator_base import GroupFeatureGeneratorBase


class MissingValueCounter(GroupFeatureGeneratorBase):
    """Counts the number of rows with missing values per group while keeping all rows."""

    def __init__(
        self,
        groupby_col: str = "artist_name",
        feature_cols: Optional[Iterable[str]] = None,
        feature_prefix: str = "ft_",
    ):
        super().__init__(groupby_col=groupby_col, feature_prefix=feature_prefix)
        self.feature_cols = list(feature_cols) if feature_cols is not None else ["popularity"]
        self._resolved_feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._resolved_feature_cols = self._resolve_feature_cols(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        if not self._resolved_feature_cols:
            self._resolved_feature_cols = self._resolve_feature_cols(X)
        feature_cols = self._resolved_feature_cols
        if not feature_cols:
            raise ValueError("No feature columns available for missing value counting.")
        result_df = X.copy()
        for col in feature_cols:
            if col in X.columns:
                result_df[f"{self.feature_prefix}{col}_missing_count"] = X.groupby(self.groupby_col)[col].transform(lambda x: x.isnull().sum())
        return result_df

    def _resolve_feature_cols(self, X: pd.DataFrame) -> List[str]:
        # No drop_cols for this use case
        return self._resolve_columns(X, self.feature_cols, drop_cols=[], numeric=False)