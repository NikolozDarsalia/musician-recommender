import pandas as pd
from abc import ABC, abstractmethod
from types import Tuple


class BaseLoader(ABC):
    @abstractmethod
    def load(self, folder_path: str) -> pd.DataFrame:
        """Loads the dataset and returns a pandas DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train-test split for interaction's dataframe
        VALID ONLY FOR INTERACTION'S DATAFRAME!"""
        raise NotImplementedError
