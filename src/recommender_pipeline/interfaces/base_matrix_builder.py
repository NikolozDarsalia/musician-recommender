from abc import ABC, abstractmethod

class BaseDatasetBuilder(ABC):
    @abstractmethod
    def fit(self):
        """Register users, items, and features."""
        pass
    
    @abstractmethod
    def build_matrices(self):
        """Return interactions, weights, user_features, item_features."""
        pass