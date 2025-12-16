import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class BaseRecommender(ABC, BaseEstimator):
    """
    Abstract base class for recommenders compatible with scikit-learn pipelines.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None, user_features=None, item_features=None):
        """
        Train the recommender on interaction data.
        X: interactions DataFrame
        y: optional target (not used in our recommenders)
        user_features: optional user side information
        item_features: optional item side information
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_ids, k=10) -> pd.DataFrame:
        """
        Return top-k recommended items per user.
        """
        raise NotImplementedError
