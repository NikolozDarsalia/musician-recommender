from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def compute(self, model, interactions, k: int):
        raise NotImplementedError
