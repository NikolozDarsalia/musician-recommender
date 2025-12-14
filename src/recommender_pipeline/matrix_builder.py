import pandas as pd
from lightfm.data import Dataset
from .interfaces.base_matrix_builder import BaseDatasetBuilder

class DatasetBuilder(BaseDatasetBuilder):
    def __init__(self, item_identity_features=True, user_identity_features=True):
        """
        Initialize the MatrixBuilder.

        Parameters
        ----------
        item_identity_features : bool, default=True
            Whether to add identity features for items (one feature per item).
            True = hybrid model (content + collaborative), False = pure content-based.
        user_identity_features : bool, default=True
            Whether to add identity features for users.
        """
        self.interactions_df = None
        self.user_features_df = None
        self.item_features_df = None
        self.dataset = Dataset(
            item_identity_features=item_identity_features,
            user_identity_features=user_identity_features
        )
        
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
            # Register all feature column names (all columns except the first which is item_id)
            item_features = list(item_features_df.columns[1:])
            # Add the no_features flag as a possible feature
            item_features.append('no_features')
        
        # Fit the LightFM dataset
        self.dataset.fit(users=users, items=items,
                         user_features=user_features,
                         item_features=item_features)
    
    def _build_user_features(self, normalize_features=True):
        """
        Build user features matrix.
        
        Parameters
        ----------
        normalize_features : bool, optional
            Whether to normalize feature matrices. Default is True.
            
        Returns
        -------
        user_features : scipy.sparse matrix or None
            User features matrix.
        """
        user_features = None
        if self.user_features_df is not None:
            user_features = self.dataset.build_user_features(
                [(row[0], [row[1]]) for row in self.user_features_df.itertuples(index=False)],
                normalize=normalize_features
            )
        return user_features
    
    def _build_item_features(self, normalize_features=True):
        """
        Build item features matrix.
        
        Parameters
        ----------
        normalize_features : bool, optional
            Whether to normalize feature matrices. Default is True.
            
        Returns
        -------
        item_features : scipy.sparse matrix or None
            Item features matrix.
        """
        item_features = None
        if self.item_features_df is not None:
            # For numeric features, create feature dictionaries
            feature_data = []
            feature_columns = self.item_features_df.columns[1:]  # All columns except first (item_id)
            
            for row in self.item_features_df.itertuples(index=False):
                item_id = row[0]
                feature_dict = {}
                has_valid_features = False
                
                # Check each feature for validity (not NaN, not zero)
                for i, col in enumerate(feature_columns):
                    value = row[i+1]
                    if pd.notna(value) and value != 0:
                        feature_dict[col] = value
                        has_valid_features = True
                
                # Add no_features flag for items without valid features
                if not has_valid_features:
                    feature_dict['no_features'] = 1.0
                
                feature_data.append((item_id, feature_dict))
            
            item_features = self.dataset.build_item_features(
                feature_data,
                normalize=normalize_features
            )
        return item_features
    
    def build_matrices(self, normalize_features=True):
        """
        Build interaction and feature matrices.
        
        Parameters
        ----------
        normalize_features : bool, optional
            Whether to normalize feature matrices. Default is True.
        """
        # Build interactions - expect columns: [user_id, item_id] or [user_id, item_id, weight]
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
        
        # Build user and item features using modular methods
        user_features = self._build_user_features(normalize_features)
        item_features = self._build_item_features(normalize_features)
        
        return interactions, weights, user_features, item_features
