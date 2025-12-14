import pandas as pd

from .group_feature_generator_base import GroupFeatureGeneratorBase


class RowCounter(GroupFeatureGeneratorBase):
    """Counts the number of rows per group while keeping all rows."""

    def __init__(
        self,
        groupby_col: str = "artist_name",
        count_col_name: str = "row_count",
        feature_prefix: str = "ft_",
    ):
        super().__init__(groupby_col=groupby_col, feature_prefix=feature_prefix)
        self.count_col_name = count_col_name

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        result_df = X.copy()
        result_df[f"{self.feature_prefix}{self.count_col_name}"] = X.groupby(self.groupby_col).transform('size')
        return result_df