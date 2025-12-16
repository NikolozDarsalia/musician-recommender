# join_listeners.py
import pandas as pd
from ...interfaces.base_preprocessor import BasePreprocessor


class JoinArtistListeners(BasePreprocessor):
    def __init__(self, artist_id="artistID", how="left"):
        self.how = how
        self.artist_id = artist_id

    def fit(self, X, y=None):
        self.listener_df_ = y
        return self

    def transform(self, spotify_df: pd.DataFrame) -> pd.DataFrame:
        return spotify_df.merge(self.listener_df_, on=self.artist_id, how=self.how)
