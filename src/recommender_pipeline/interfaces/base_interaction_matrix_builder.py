import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BaseMatrixBuilder(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract base class for matrix builders compatible with scikit-learn pipelines.
    Converts interaction data into a user-item matrix.
    """

    def __init__(self, users=None, items=None):
        """
        Parameters
        ----------
        users : list or array-like, optional
            List of user IDs to include in the matrix.
        items : list or array-like, optional
            List of item IDs to include in the matrix.
        """
        self.users = users
        self.items = items

    def fit(self, X: pd.DataFrame, y=None) -> "BaseMatrixBuilder":
        """
        Fit to data. Stores users/items if not provided.
        """
        if self.users is None:
            self.users = X["user_id"].unique()
        if self.items is None:
            self.items = X["item_id"].unique()
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the interactions DataFrame into a user-item matrix.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement transform()")

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Default implementation: fit then transform.
        """
        return self.fit(X, y).transform(X)
