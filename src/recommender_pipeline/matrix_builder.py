import pandas as pd
from lightfm.data import Dataset
from .interfaces.base_matrix_builder import BaseDatasetBuilder

class DatasetBuilder(BaseDatasetBuilder):
    def __init__(self):
        """
        Initialize the MatrixBuilder.

        This constructor sets up the MatrixBuilder instance for subsequent
        matrix construction operations. Data should be provided via the fit() method.
        """
        self.interactions_df = None
        self.user_features_df = None
        self.item_features_df = None
        self.dataset = Dataset()
        
    def fit(self, interactions_df, user_features_df=None, item_features_df=None):
        """
        Fit the dataset with user/item mappings and optional feature mappings.
        
        Parameters
        ----------
        interactions_df : pandas.DataFrame
            A dataframe containing user-item interaction data. Expected columns in order:
            [user_id, item_id] or [user_id, item_id, weight]
        user_features_df : pandas.DataFrame, optional
            A dataframe containing features for users. Expected columns in order:
            [user_id, feature] or with additional feature columns
        item_features_df : pandas.DataFrame, optional
            A dataframe containing features for items. Expected columns in order:
            [item_id, feature] or with additional feature columns
        """
        # Store the dataframes for use in build_matrices
        self.interactions_df = interactions_df
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        
        users = interactions_df.iloc[:, 0].unique()  # First column: user_id
        items = interactions_df.iloc[:, 1].unique()  # Second column: item_id
        
        # Include items from item_features_df if provided (for cold start items)
        if item_features_df is not None:
            feature_items = item_features_df.iloc[:, 0].unique()  # First column: item_id
            items = pd.Series(list(set(items) | set(feature_items))).values  # Union of both sets
        
        # Store as attributes for inspection
        self.users = users
        self.items = items
        
        user_features = []
        if user_features_df is not None:
            user_features = user_features_df.iloc[:, 1].unique()  # Second column: feature
        
        item_features = []
        if item_features_df is not None:
            item_features = item_features_df.iloc[:, 1].unique()  # Second column: feature
        
        # Fit the LightFM dataset
        self.dataset.fit(users=users, items=items,
                         user_features=user_features,
                         item_features=item_features)
    
    def build_matrices(self, normalize_features=True):
        """
        Build interaction and feature matrices.
        
        Parameters
        ----------
        normalize_features : bool, optional
            Whether to normalize feature matrices. Default is True.
        """
        # Interactions - expect columns: [user_id, item_id] or [user_id, item_id, weight]
        if self.interactions_df.shape[1] >= 3:
            # Has weight column
            interactions, weights = self.dataset.build_interactions(
                [(row[0], row[1], row[2]) for row in self.interactions_df.itertuples(index=False)]
            )
        else:
            # No weight column, default weight=1.0
            interactions, weights = self.dataset.build_interactions(
                [(row[0], row[1]) for row in self.interactions_df.itertuples(index=False)]
            )
        
        # User features - expect columns: [user_id, feature, ...]
        user_features = None
        if self.user_features_df is not None:
            user_features = self.dataset.build_user_features(
                [(row[0], [row[1]]) for row in self.user_features_df.itertuples(index=False)],
                normalize=normalize_features
            )
        
        # Item features - expect columns: [item_id, feature, ...]
        item_features = None
        if self.item_features_df is not None:
            item_features = self.dataset.build_item_features(
                [(row[0], [row[1]]) for row in self.item_features_df.itertuples(index=False)],
                normalize=normalize_features
            )
        
        return interactions, weights, user_features, item_features
