import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureGenerator(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract base class for feature generators compatible with scikit-learn pipelines.
    """

    def __init__(self):
        # Add any parameters you want to configure here
        pass

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit to data. For feature generators, this often does nothing,
        but we can compute statistics if needed.
        """
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input DataFrame into feature matrix.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement transform()")

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Optional: scikit-learn already provides a default implementation,
        but you can override if more efficient.
        """
        return self.fit(X, y).transform(X)
