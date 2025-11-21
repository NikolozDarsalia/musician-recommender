from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, interactions_df, user_features=None, item_features=None):
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_ids, k):
        """Returns ranked list of top-k items per user."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """Serialize model + metadata."""

    @abstractmethod
    def load(self, path: str):
        """Restore model from disk."""
