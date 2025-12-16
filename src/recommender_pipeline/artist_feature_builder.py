import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import Dict


class ArtistFeaturesBuilder:
    """
    Builds artist feature matrix for LightFM from metadata.

    Key features:
    - Converts artist metadata to sparse feature matrix
    - Aligns features with artist mappings from InteractionMatrixBuilder
    - Handles missing artists gracefully
    """

    def __init__(self):
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(
        self, spotify_new: pd.DataFrame, artist_id_map: Dict
    ) -> csr_matrix:
        """
        Build artist feature matrix aligned with artist mappings.

        Args:
            spotify_new: DataFrame with artistID as index and scaled feature columns
            artist_id_map: Artist ID to matrix index mapping from InteractionMatrixBuilder

        Returns:
            Sparse CSR matrix of shape (n_artists, n_features)
        """
        n_artists = len(artist_id_map)
        n_features = spotify_new.shape[1]

        self.feature_names = list(spotify_new.columns)

        # Build sparse matrix
        rows = []
        cols = []
        data = []

        for artistID in spotify_new.index:
            if artistID in artist_id_map:
                artist_idx = artist_id_map[artistID]
                feature_values = spotify_new.loc[artistID].values

                for feature_idx, value in enumerate(feature_values):
                    if value != 0:  # Only store non-zero values for sparsity
                        rows.append(artist_idx)
                        cols.append(feature_idx)
                        data.append(value)

        features_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_artists, n_features)
        )

        self.is_fitted = True
        return features_matrix.tocsr()

    def get_feature_vector(
        self, artistID: str, spotify_new: pd.DataFrame
    ) -> np.ndarray:
        """
        Get feature vector for a specific artist.
        Useful for analyzing what features drive recommendations.
        """
        if artistID in spotify_new.index:
            return spotify_new.loc[artistID].values
        else:
            return np.zeros(len(self.feature_names))
