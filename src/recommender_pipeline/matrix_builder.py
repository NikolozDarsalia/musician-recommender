import pandas as pd
from lightfm.data import Dataset
from .interfaces.base_matrix_builder import BaseDatasetBuilder

class DatasetBuilder(BaseDatasetBuilder):
    def __init__(self, interactions_df, user_features_df=None, item_features_df=None):
        """
        Initialize the MatrixBuilder with interaction data and optional feature data.

        This constructor sets up the MatrixBuilder instance with the necessary dataframes
        for building recommendation matrices. It prepares the dataset object for subsequent
        matrix construction operations.

        Parameters
        ----------
        interactions_df : pandas.DataFrame
            A dataframe containing user-item interaction data. Typically includes columns
            for user IDs, item IDs, and interaction values (e.g., ratings, play counts).
        user_features_df : pandas.DataFrame, optional
            A dataframe containing additional features for users. Default is None.
            If provided, should include user IDs and their associated feature columns.
        item_features_df : pandas.DataFrame, optional
            A dataframe containing additional features for items. Default is None.
            If provided, should include item IDs and their associated feature columns.
        """
        self.interactions_df = interactions_df
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.dataset = Dataset()
        
    def fit(self):
        users = self.interactions_df['user_id'].unique()
        items = self.interactions_df['item_id'].unique()
        
        user_features = []
        if self.user_features_df is not None:
            user_features = self.user_features_df['feature'].unique()
        
        item_features = []
        if self.item_features_df is not None:
            item_features = self.item_features_df['feature'].unique()
        
        self.dataset.fit(users=users, items=items,
                         user_features=user_features,
                         item_features=item_features)
    
    def build_matrices(self):
        # Interactions
        interactions, weights = self.dataset.build_interactions(
            [(row.user_id, row.item_id, row.weight) for row in self.interactions_df.itertuples()]
        )
        
        # User features
        user_features = None
        if self.user_features_df is not None:
            user_features = self.dataset.build_user_features(
                [(row.user_id, [row.feature]) for row in self.user_features_df.itertuples()]
            )
        
        # Item features
        item_features = None
        if self.item_features_df is not None:
            item_features = self.dataset.build_item_features(
                [(row.item_id, [row.feature]) for row in self.item_features_df.itertuples()]
            )
        
        return interactions, weights, user_features, item_features
