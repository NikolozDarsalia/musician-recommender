import pandas as pd
from abc import ABC, abstractmethod

class BaseLoader(ABC):

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Loads the dataset and returns a pandas DataFrame."""
        raise NotImplementedError
