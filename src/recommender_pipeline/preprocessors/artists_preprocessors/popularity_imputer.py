from sklearn.preprocessing import MinMaxScaler
from ...interfaces.base_preprocessor import BasePreprocessor
import pandas as pd


class PopularityImputer(BasePreprocessor):
    def __init__(
        self,
        popularity_col="popularity",
        listeners_col="num_listeners",
    ):
        self.popularity_col = popularity_col
        self.listeners_col = listeners_col
        self.scaler_ = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y=None):
        mask = X[self.popularity_col].notnull() & X[self.listeners_col].notnull()
        self.scaler_.fit(X.loc[mask, [self.listeners_col]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        scaled_listeners = (
            self.scaler_.transform(X[[self.listeners_col]]).flatten() * 100
        )

        fill_mask = X[self.popularity_col].isnull() & X[self.listeners_col].notnull()
        X.loc[fill_mask, self.popularity_col] = scaled_listeners[fill_mask]

        return X
