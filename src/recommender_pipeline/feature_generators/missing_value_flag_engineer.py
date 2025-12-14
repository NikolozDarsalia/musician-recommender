from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from ..interfaces.base_feature_generator import BaseFeatureGenerator


class MissingValueFlagEngineer(BaseFeatureGenerator):
    """
    Creates binary flag features indicating missing values for each column.

    For each column with missing values, creates a new column '{col}_is_missing'
    with 1 if the value is missing, 0 otherwise.
    """

    def __init__(
        self,
        feature_cols: Optional[Iterable[str]] = None,
        exclude_cols: Optional[Iterable[str]] = None,
        min_missing_rate: float = 0.0,
        max_missing_rate: float = 1.0,
        suffix: str = "_is_missing",
        dtype: Union[type, str] = "int8",
        include_missing_count: bool = False,
        missing_count_col: str = "n_missing_features",
    ) -> None:
        """
        Parameters
        ----------
        feature_cols : Optional[Iterable[str]]
            Explicit list of columns to check for missing values. If None, all columns
            with at least one missing value are used.
        exclude_cols : Optional[Iterable[str]]
            Columns to exclude from missing value flag generation (e.g., ID columns).
        min_missing_rate : float, default=0.0
            Minimum fraction of missing values (0-1) a column must have to create a flag.
            Use 0.01 to ignore columns with <1% missing values.
        max_missing_rate : float, default=1.0
            Maximum fraction of missing values (0-1) a column can have to create a flag.
            Use 0.99 to ignore columns that are almost entirely missing.
        suffix : str, default="_is_missing"
            Suffix to append to original column names for flag columns.
        dtype : Union[type, str], default="int8"
            Data type for flag columns (int8 saves memory).
        include_missing_count : bool, default=False
            If True, adds a column counting total missing features per row.
        missing_count_col : str, default="n_missing_features"
            Name for the total missing count column (if include_missing_count=True).
        """
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.exclude_cols = set(exclude_cols) if exclude_cols is not None else set()
        self.min_missing_rate = min_missing_rate
        self.max_missing_rate = max_missing_rate
        self.suffix = suffix
        self.dtype = dtype
        self.include_missing_count = include_missing_count
        self.missing_count_col = missing_count_col

        # Will be set during fit
        self.cols_with_missing_: List[str] = []
        self.missing_rates_: dict = {}
        self.n_features_in_: int = 0

    def fit(self, X: pd.DataFrame, y=None):
        """
        Identify columns with missing values that meet the criteria.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to analyze for missing values.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingValueFlagEngineer.fit expects a DataFrame.")

        self.n_features_in_ = X.shape[1]

        # Determine which columns to check
        if self.feature_cols is not None:
            cols_to_check = [col for col in self.feature_cols if col in X.columns]
            missing = set(self.feature_cols) - set(X.columns)
            if missing:
                raise KeyError(
                    f"Feature columns not found in DataFrame: {sorted(missing)}"
                )
        else:
            cols_to_check = list(X.columns)

        # Remove excluded columns
        cols_to_check = [col for col in cols_to_check if col not in self.exclude_cols]

        # Find columns with missing values
        self.cols_with_missing_ = []
        self.missing_rates_ = {}

        for col in cols_to_check:
            n_missing = X[col].isna().sum()

            if n_missing > 0:
                missing_rate = n_missing / len(X)

                # Check if missing rate is within bounds
                if self.min_missing_rate <= missing_rate <= self.max_missing_rate:
                    self.cols_with_missing_.append(col)
                    self.missing_rates_[col] = missing_rate

        # Sort for consistency
        self.cols_with_missing_ = sorted(self.cols_with_missing_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary flag features for missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            DataFrame with binary flag columns for each column with missing values.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingValueFlagEngineer.transform expects a DataFrame.")

        if not hasattr(self, "cols_with_missing_"):
            raise RuntimeError(
                "MissingValueFlagEngineer must be fitted before transform."
            )

        if len(self.cols_with_missing_) == 0:
            # No missing values found during fit
            if self.include_missing_count:
                return pd.DataFrame(
                    {self.missing_count_col: np.zeros(len(X), dtype=self.dtype)},
                    index=X.index,
                )
            else:
                return pd.DataFrame(index=X.index)

        # Create flag columns
        flag_data = {}

        for col in self.cols_with_missing_:
            if col not in X.columns:
                raise KeyError(
                    f"Column '{col}' was present during fit but missing in transform."
                )

            flag_col_name = f"{col}{self.suffix}"
            flag_data[flag_col_name] = X[col].isna().astype(self.dtype)

        flags_df = pd.DataFrame(flag_data, index=X.index)

        # Add total missing count if requested
        if self.include_missing_count:
            flags_df[self.missing_count_col] = flags_df.sum(axis=1).astype(self.dtype)

        return flags_df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit to data and transform it in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        pd.DataFrame
            DataFrame with binary flag columns.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : Ignored
            Not used, present for API consistency.

        Returns
        -------
        List[str]
            List of output feature names.
        """
        if not hasattr(self, "cols_with_missing_"):
            raise RuntimeError(
                "MissingValueFlagEngineer must be fitted before get_feature_names_out."
            )

        feature_names = [f"{col}{self.suffix}" for col in self.cols_with_missing_]

        if self.include_missing_count:
            feature_names.append(self.missing_count_col)

        return feature_names

    def get_missing_summary(self) -> pd.DataFrame:
        """
        Get summary statistics about missing values in fitted data.

        Returns
        -------
        pd.DataFrame
            Summary with columns: 'column', 'missing_rate', 'flag_created'
        """
        if not hasattr(self, "cols_with_missing_"):
            raise RuntimeError(
                "MissingValueFlagEngineer must be fitted before get_missing_summary."
            )

        summary_data = []

        for col, rate in self.missing_rates_.items():
            summary_data.append(
                {"column": col, "missing_rate": rate, "flag_created": True}
            )

        return (
            pd.DataFrame(summary_data)
            .sort_values("missing_rate", ascending=False)
            .reset_index(drop=True)
        )
