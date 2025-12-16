from .interfaces.base_loader import BaseLoader
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class StandardLoader(BaseLoader):
    """Loader for all dataset"""

    def __init__(self, folder_path: str = None):
        self.folder_path = folder_path

    def load(self) -> pd.DataFrame:
        # returns df with columns from artists_spotify_matched.parquet

        file_format = self.folder_path.rsplit(".", 1)[-1]
        if file_format == "csv":
            df = pd.read_csv(self.folder_path)
        elif file_format == "dat":
            df = pd.read_csv(self.folder_path, sep="\t", encoding="latin1")
        elif file_format == "parquet":
            df = pd.read_parquet(self.folder_path)
        elif file_format == "pickle" or file_format == "pkl":
            df = pd.read_pickle(self.folder_path)
        else:
            raise Exception(f"Unsupported file format - {file_format}")

        return df

    def train_test_val_split(
        self,
        df: pd.DataFrame,
        strategy: str = "random",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train/val/test using a strategy.
        """
        if strategy == "random":
            return self._random_split(df, test_size, val_size, random_state)

        elif strategy == "user_stratified":
            return self._user_stratified(df, test_size, val_size, random_state)

    def _random_split(self, df, test_size, val_size, random_state):
        # First split off test set
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        # Now split train_val into train and validation
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),  # adjust proportion
            random_state=random_state,
        )
        return train, val, test

    def _user_stratified(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Stratified split by userID, with safeguards for users with few interactions.
        Ensures each user appears in train/val/test if possible, otherwise falls back gracefully.
        """
        users = df["userID"].unique()
        train_parts, val_parts, test_parts = [], [], []

        for u in users:
            subset = df[df["userID"] == u]

            # If user has fewer than 3 interactions, put all in train
            if len(subset) < 3:
                train_parts.append(subset)
                continue

            # First split off test
            try:
                train_val_u, test_u = train_test_split(
                    subset, test_size=test_size, random_state=random_state
                )
            except ValueError:
                # Not enough samples → put all in train
                train_parts.append(subset)
                continue

            # Then split train_val into train and val
            try:
                train_u, val_u = train_test_split(
                    train_val_u,
                    test_size=val_size / (1 - test_size),
                    random_state=random_state,
                )
            except ValueError:
                # Not enough samples → put all in train
                train_u, val_u = train_val_u, pd.DataFrame()

            train_parts.append(train_u)
            val_parts.append(val_u)
            test_parts.append(test_u)

        # Concatenate safely (skip empty lists)
        train_df = pd.concat(train_parts) if train_parts else pd.DataFrame()
        val_df = pd.concat(val_parts) if val_parts else pd.DataFrame()
        test_df = pd.concat(test_parts) if test_parts else pd.DataFrame()

        return train_df, val_df, test_df
