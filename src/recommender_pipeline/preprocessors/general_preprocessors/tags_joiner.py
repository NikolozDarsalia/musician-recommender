# join_tags.py
import pandas as pd
from ...interfaces.base_preprocessor import BasePreprocessor


class JoinTags(BasePreprocessor):
    """
    Join aggregated artist tags to a target dataset.
    """

    def __init__(self, id="artistID", how="left"):
        self.how = how
        self.id = id

    def fit(self, X, y=None):
        self.tag_df_ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.merge(self.tag_df_, on=self.id, how=self.how)
