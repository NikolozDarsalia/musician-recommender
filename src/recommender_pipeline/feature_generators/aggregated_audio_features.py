from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from ..interfaces.base_feature_generator import BaseFeatureGenerator


AggFunc = Union[str, Callable[[pd.Series], float]]


class AggregatedAudioGenerator(BaseFeatureGenerator):
    """Aggregates track-level audio features to the artist level."""

    def __init__(
        self,
        groupby_col: str = "artist_name",
        feature_cols: Optional[Iterable[str]] = None,
        genre_col: Optional[str] = "track_genre",
        agg_funcs: Sequence[AggFunc] = ("mean", "median", "std", "min", "max"),
        track_count_col: str = "track_count",
    ) -> None:
        """
        Parameters
        ----------
        groupby_col:
            Column name used to identify the artist.
        feature_cols:
            Explicit list of numeric feature columns to aggregate. If None, all numeric
            columns except the grouping/genre columns are used.
        genre_col:
            Name of the genre column. When present, top genre and unique genre counts
            are added.
        agg_funcs:
            Aggregations to apply to each numeric feature column.
        track_count_col:
            Name for the column that stores the number of tracks per artist.
        """
        self.groupby_col = groupby_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.genre_col = genre_col
        self.agg_funcs = list(agg_funcs)
        self.track_count_col = track_count_col
        self._resolved_feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._resolved_feature_cols = self._resolve_feature_cols(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AggregatedAudioGenerator.transform expects a DataFrame.")

        if self.groupby_col not in X.columns:
            raise KeyError(f"Missing grouping column '{self.groupby_col}'.")

        feature_cols = self._resolved_feature_cols or self._resolve_feature_cols(X)
        if not feature_cols:
            raise ValueError("No feature columns available for aggregation.")

        grouped = X.groupby(self.groupby_col)
        agg_dict: Dict[str, Sequence[AggFunc]] = {col: self.agg_funcs for col in feature_cols}

        agg_df = grouped.agg(agg_dict)
        agg_df.columns = [
            self._format_agg_column(col_name, agg_name) for col_name, agg_name in agg_df.columns
        ]

        agg_df[self.track_count_col] = grouped.size()

        if self.genre_col and self.genre_col in X.columns:
            agg_df[f"{self.genre_col}_top"] = grouped[self.genre_col].agg(self._mode_safe)
            agg_df[f"{self.genre_col}_n_unique"] = grouped[self.genre_col].nunique(dropna=True)

        return agg_df.reset_index()

    def _resolve_feature_cols(self, X: pd.DataFrame) -> List[str]:
        if self.feature_cols is not None:
            missing = set(self.feature_cols) - set(X.columns)
            if missing:
                raise KeyError(f"Feature columns not found in DataFrame: {sorted(missing)}")
            return list(self.feature_cols)

        drop_cols = [c for c in [self.groupby_col, self.genre_col] if c in X.columns]
        numeric_cols = X.drop(columns=drop_cols).select_dtypes(include=["number"]).columns
        return [col for col in numeric_cols if not str(col).startswith("__")]

    @staticmethod
    def _mode_safe(series: pd.Series):
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else None

    @staticmethod
    def _format_agg_column(col: str, agg_name: Union[str, Callable]) -> str:
        if isinstance(agg_name, str):
            suffix = agg_name
        elif hasattr(agg_name, "__name__") and agg_name.__name__ != "<lambda>":
            suffix = agg_name.__name__
        else:
            suffix = "agg"
        return f"{col}_{suffix}"
