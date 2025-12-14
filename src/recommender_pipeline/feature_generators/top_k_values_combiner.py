from typing import Iterable, List, Optional

import pandas as pd

from .group_feature_generator_base import GroupFeatureGeneratorBase


class TopKValuesCombiner(GroupFeatureGeneratorBase):
    """Combines top k unique values into strings for categorical columns while keeping all rows."""

    def __init__(
        self,
        groupby_col: str = "artist_name",
        categorical_feature_cols: Optional[Iterable[str]] = None,
        k: int = 5,
        separator: str = " | ",
        feature_prefix: str = "ft_",
    ):
        super().__init__(groupby_col=groupby_col, feature_prefix=feature_prefix)
        self.categorical_feature_cols = list(categorical_feature_cols) if categorical_feature_cols is not None else None
        self.k = k
        self.separator = separator
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
            raise ValueError("No categorical columns available for top k values combination.")
        result_df = X.copy()
        for col in categorical_cols:
            if col in X.columns:
                result_df[f"{self.feature_prefix}{col}_top_{self.k}_combined"] = X.groupby(self.groupby_col)[col].transform(self._get_top_k_combined)
        return result_df

    def _resolve_categorical_cols(self, X: pd.DataFrame) -> List[str]:
        drop_cols = [self.groupby_col] if self.groupby_col in X.columns else []
        return self._resolve_columns(X, self.categorical_feature_cols, drop_cols, numeric=False)

    def _get_top_k_combined(self, series: pd.Series) -> str:
        """Get top k most frequent values combined into a string."""
        if series.empty:
            return ""
        
        # Get value counts and take top k
        value_counts = series.value_counts(dropna=True)
        top_k_values = value_counts.head(self.k).index.tolist()
        
        # Convert to strings and join
        top_k_strings = [str(val) for val in top_k_values]
        return self.separator.join(top_k_strings)