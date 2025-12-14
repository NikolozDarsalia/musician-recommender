from typing import Iterable, List, Optional

import pandas as pd

from .group_feature_generator_base import GroupFeatureGeneratorBase


class UniqueCounter(GroupFeatureGeneratorBase):
    """Counts unique values for categorical columns while keeping all rows."""

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
            raise ValueError("No categorical columns available for unique counting.")
        result_df = X.copy()
        for col in categorical_cols:
            if col in X.columns:
                result_df[f"{self.feature_prefix}{col}_n_unique"] = X.groupby(self.groupby_col)[col].transform('nunique')
        return result_df

    def _resolve_categorical_cols(self, X: pd.DataFrame) -> List[str]:
        drop_cols = [self.groupby_col] if self.groupby_col in X.columns else []
        return self._resolve_columns(X, self.categorical_feature_cols, drop_cols, numeric=False)