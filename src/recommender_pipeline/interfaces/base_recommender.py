from abc import ABC, abstractmethod

class BaseRecommender(ABC):

    @abstractmethod
    def fit(self, interactions, user_features=None, item_features=None):
        raise NotImplementedError
    
    @abstractmethod
    def recommend(self, user_ids, k=10):
        """Returns ranked list of top-k items per user."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Serialize model + metadata."""
    
    @classmethod
    def load(cls, path: str):
        """Restore model from disk."""
