from abc import ABC, abstractmethod

class BaseTuner(ABC):

    @abstractmethod
    def optimize(self, train, val, user_features, item_features):
        """Returns best hyperparameters."""
        raise NotImplementedError
