from ..interfaces.base_feature_generator import BaseFeatureGenerator

class KeepGroupFeats(BaseFeatureGenerator):
    """
    Transformer that drops specified original features from a DataFrame.
    Keeps only the transformed features at groupby level.
    Ensures group_col is not dropped even if present in original_features.
    """
    def __init__(self, original_features, group_col="artist_name"):
        self.original_features = original_features
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure group_col is not dropped
        drop_cols = [col for col in self.original_features if col != self.group_col]
        return X.drop(columns=drop_cols).drop_duplicates(subset=self.group_col).reset_index(drop=True)