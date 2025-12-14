from typing import Callable, Iterable, List, Optional, Sequence, Union

import pandas as pd
import numpy as np


from .group_feature_generator_base import GroupFeatureGeneratorBase


AggFunc = Union[str, Callable[[pd.Series], float]]


class NumericalAggregator(GroupFeatureGeneratorBase):
    """Generates aggregated numerical features for each group while keeping all rows."""

    def __init__(
        self,
        groupby_col: str = "artist_name",
        numerical_feature_cols: Optional[Iterable[str]] = None,
        agg_funcs: Sequence[AggFunc] = ("mean", "median", "std", "min", "max"),
        feature_prefix: str = "ft_",
    ):
        super().__init__(groupby_col=groupby_col, feature_prefix=feature_prefix)
        self.numerical_feature_cols = list(numerical_feature_cols) if numerical_feature_cols is not None else None
        self.agg_funcs = list(agg_funcs)
        self._resolved_numerical_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._resolved_numerical_cols = self._resolve_numerical_cols(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        if not self._resolved_numerical_cols:
            self._resolved_numerical_cols = self._resolve_numerical_cols(X)
        numerical_cols = self._resolved_numerical_cols
        if not numerical_cols:
            raise ValueError("No numerical columns available for aggregation.")
        result_df = X.copy()
        for col in numerical_cols:
            for agg_func in self.agg_funcs:
                new_col_name = f"{self.feature_prefix}{self._format_agg_column(col, agg_func)}"
                result_df[new_col_name] = X.groupby(self.groupby_col)[col].transform(agg_func)
        return result_df

    def _resolve_numerical_cols(self, X: pd.DataFrame) -> List[str]:
        drop_cols = [self.groupby_col] if self.groupby_col in X.columns else []
        return self._resolve_columns(X, self.numerical_feature_cols, drop_cols, numeric=True)

    @staticmethod
    def _format_agg_column(col: str, agg_name: Union[str, Callable]) -> str:
        """Format aggregated column name."""
        if isinstance(agg_name, str):
            suffix = agg_name
        elif hasattr(agg_name, "__name__") and agg_name.__name__ != "<lambda>":
            suffix = agg_name.__name__
        else:
            suffix = "agg"
        return f"{col}_{suffix}"