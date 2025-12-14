from typing import Iterable, List, Optional
import pandas as pd
from ..interfaces.base_feature_generator import BaseFeatureGenerator

class GroupFeatureGeneratorBase(BaseFeatureGenerator):
    """
    Base class for group-based feature generators.
    Handles groupby_col, feature_prefix, and column resolution logic.
    """
    def __init__(
        self,
        groupby_col: str = "artist_name",
        feature_prefix: str = "ft_",
    ):
        self.groupby_col = groupby_col
        self.feature_prefix = feature_prefix

    def _check_input(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.transform expects a DataFrame.")
        if self.groupby_col not in X.columns:
            raise KeyError(f"Missing grouping column '{self.groupby_col}'.")

    def _copy_and_resolve(self, X: pd.DataFrame, resolved_cols: Optional[List[str]], resolve_func):
        # Resolve feature columns if not already done
        if not resolved_cols:
            resolved_cols = resolve_func(X)
        result_df = X.copy()
        return result_df, resolved_cols

    def _resolve_columns(self, X: pd.DataFrame, user_cols: Optional[Iterable[str]], drop_cols: Iterable[str], numeric: bool = False) -> List[str]:
        if user_cols is not None:
            missing = set(user_cols) - set(X.columns)
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {sorted(missing)}")
            return list(user_cols)
        else:
            available_cols = X.drop(columns=drop_cols).columns
            if numeric:
                cols = X.drop(columns=drop_cols).select_dtypes(include=["number"]).columns
                return [col for col in cols if not str(col).startswith("__")]
            else:
                numeric_cols = X.drop(columns=drop_cols).select_dtypes(include=["number"]).columns
                return [col for col in available_cols if col not in numeric_cols and not str(col).startswith("__")]
