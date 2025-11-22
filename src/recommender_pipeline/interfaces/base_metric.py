from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, model, interactions_df, user_features=None, item_features=None):
        """
        Compute the metric on the given interactions matrix.
        Returns a float.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """Return the metric name."""
        pass
