from ...interfaces.base_preprocessor import BasePreprocessor
import pandas as pd


class HighNullDropper(BasePreprocessor):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X: pd.DataFrame, y=None):
        null_frac = X.isnull().mean()
        self.cols_to_drop_ = null_frac[null_frac > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")
