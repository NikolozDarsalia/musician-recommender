import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import Dict


class InteractionMatrixBuilder:
    """
    Builds interaction matrices for LightFM with support for cold-start users.

    Key features:
    - Maintains consistent user/artist mappings across train/test/val
    - Supports adding new users dynamically without retraining
    - Handles cold-start scenario through survey-based interactions
    """

    def __init__(self):
        self.user_id_map: Dict = {}
        self.artist_id_map: Dict = {}
        self.reverse_user_map: Dict = {}
        self.reverse_artist_map: Dict = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Create mappings from all data splits.

        Args:
            train_df, test_df, val_df: DataFrames with ['userID', 'artistID', 'weight']
        """
        # Collect all unique users and artists
        all_users = pd.concat(
            [train_df["userID"], test_df["userID"], val_df["userID"]]
        ).unique()

        all_artists = pd.concat(
            [train_df["artistID"], test_df["artistID"], val_df["artistID"]]
        ).unique()

        # Create bidirectional mappings
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(all_users)}
        self.artist_id_map = {
            artist_id: idx for idx, artist_id in enumerate(all_artists)
        }

        self.reverse_user_map = {
            idx: user_id for user_id, idx in self.user_id_map.items()
        }
        self.reverse_artist_map = {
            idx: artist_id for artist_id, idx in self.artist_id_map.items()
        }

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Convert interaction DataFrame to sparse matrix.

        Args:
            df: DataFrame with ['userID', 'artistID', 'weight']

        Returns:
            Sparse CSR matrix of shape (n_users, n_artists)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")

        # Map IDs to indices
        user_indices = df["userID"].map(self.user_id_map)
        artist_indices = df["artistID"].map(self.artist_id_map)
        values = df["weight"].values

        # Filter out unmapped items (shouldn't happen if fitted properly)
        valid_mask = user_indices.notna() & artist_indices.notna()
        user_indices = user_indices[valid_mask].astype(int)
        artist_indices = artist_indices[valid_mask].astype(int)
        values = values[valid_mask]

        # Create sparse matrix
        matrix = coo_matrix(
            (values, (user_indices, artist_indices)),
            shape=(len(self.user_id_map), len(self.artist_id_map)),
        )

        return matrix.tocsr()

    def add_new_user_interactions(
        self, userID: str, survey_interactions: pd.DataFrame
    ) -> np.ndarray:
        """
        Create interaction vector for a NEW user based on survey responses.
        This allows predictions for cold-start users without retraining.

        Args:
            userID: New user's ID (not in training data)
            survey_interactions: DataFrame with ['artistID', 'weight']
                                Survey responses indicating user preferences

        Returns:
            Dense array of shape (n_artists,) representing user's preferences
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before adding new users")

        # Initialize empty preference vector
        user_vector = np.zeros(len(self.artist_id_map))

        # Fill in survey responses
        for _, row in survey_interactions.iterrows():
            if row["artistID"] in self.artist_id_map:
                artist_idx = self.artist_id_map[row["artistID"]]
                user_vector[artist_idx] = row["weight"]

        return user_vector

    def get_artist_indices(self, artist_ids: list) -> np.ndarray:
        """Get matrix indices for a list of artist IDs"""
        return np.array([self.artist_id_map.get(aid, -1) for aid in artist_ids])

    @property
    def n_users(self) -> int:
        return len(self.user_id_map)

    @property
    def n_artists(self) -> int:
        return len(self.artist_id_map)
