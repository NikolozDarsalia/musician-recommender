from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, model, interactions_df, k: int):
        raise NotImplementedError
