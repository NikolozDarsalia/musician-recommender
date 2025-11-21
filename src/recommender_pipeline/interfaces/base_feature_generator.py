import pandas as pd
from abc import ABC, abstractmethod

class BaseFeatureGenerator(ABC):

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a cleaned dataframe and returns a feature matrix.
        """
        raise NotImplementedError
