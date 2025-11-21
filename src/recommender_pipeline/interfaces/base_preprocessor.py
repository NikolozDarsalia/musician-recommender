import pandas as pd
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        """Optional: learn parameters (e.g., mean, mappings)."""
        return self
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset and return cleaned version."""
        raise NotImplementedError
