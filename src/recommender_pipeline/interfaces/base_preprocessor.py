import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract base class for preprocessors compatible with scikit-learn pipelines.
    """

    def __init__(self):
        # Add any configurable parameters here
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "BasePreprocessor":
        """
        Fit to data. For preprocessors, this often does nothing,
        but we can compute statistics (e.g., means, mappings).
        """
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset and return cleaned version.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement transform()")

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Optional: scikit-learn already provides a default implementation,
        but you can override if more efficient.
        """
        return self.fit(X, y).transform(X)
