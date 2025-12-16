# listener_counter.py
import pandas as pd
from ...interfaces.base_preprocessor import BasePreprocessor


class ArtistListenerCounter(BasePreprocessor):
    """
    Compute number of unique listeners.
    """

    def __init__(self, user_id="userID", artist_id="artistID"):
        self.user_id = user_id
        self.artist_id = artist_id

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        return (
            interactions.groupby(self.artist_id)
            .agg(num_listeners=(self.user_id, "nunique"))
            .reset_index()
        )
