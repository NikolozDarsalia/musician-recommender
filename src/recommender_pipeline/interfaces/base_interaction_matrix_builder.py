from abc import ABC, abstractmethod
import pandas as pd


class BaseMatrixBuilder(ABC):
    @abstractmethod
    def build(self, interactions_df: pd.Dataframe, users, items) -> pd.DataFrame:
        """
        Returns:
          interactions       -> sparse matrix [n_users x n_items]
        """
        raise NotImplementedError
