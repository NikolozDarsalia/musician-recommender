from ..interfaces.base_preprocessor import BasePreprocessor
import pandas as pd
import numpy as np


class UserMinMaxListeningScaler(BasePreprocessor):
    """
    Scales user listening counts ("weight") using Min-Max normalization
    independently for each user.

    For each user u:
        weight_scaled = (weight - min_u) / (max_u - min_u)

    Special case:
    - If a user has only one interaction (min == max),
      the scaled weight is set to 1.0.

    Output:
    -------
    DataFrame with columns:
        - userID
        - artistID
        - weight (scaled)
    """

    def __init__(
        self,
        user_col="userID",
        item_col="artistID",
        weight_col="weight",
        single_value_fill=1.0,
    ):
        """
        Args:
            user_col:
                Column identifying users.
            item_col:
                Column identifying items (artists).
            weight_col:
                Column containing number of listenings.
            single_value_fill:
                Value assigned when min == max for a user.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.weight_col = weight_col
        self.single_value_fill = single_value_fill

        # Fitted attributes
        self.user_stats_ = None  # user -> (min, max)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y=None):
        """
        Compute per-user min and max listening counts.
        """
        required_cols = {self.user_col, self.item_col, self.weight_col}
        missing = required_cols - set(X.columns)

        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Compute per-user min/max
        stats = (
            X.groupby(self.user_col)[self.weight_col]
            .agg(["min", "max"])
            .rename(columns={"min": "min_w", "max": "max_w"})
        )

        self.user_stats_ = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply per-user Min-Max scaling to listening counts.
        """
        if self.user_stats_ is None:
            raise RuntimeError("Must call fit() before transform()")

        X = X.copy()

        # Join user stats
        X = X.merge(
            self.user_stats_,
            left_on=self.user_col,
            right_index=True,
            how="left",
        )

        if X[["min_w", "max_w"]].isna().any().any():
            raise ValueError("Transform data contains unseen users")

        # Min-Max scaling per user
        denom = X["max_w"] - X["min_w"]

        X[self.weight_col] = np.where(
            denom > 0,
            (X[self.weight_col] - X["min_w"]) / denom,
            self.single_value_fill,
        )

        # Return only required columns
        return X[[self.user_col, self.item_col, self.weight_col]]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
