from ...interfaces.base_preprocessor import BasePreprocessor
import pandas as pd


class MostPopularTrackSelector(BasePreprocessor):
    """
    For each artist:
    - If popularity exists → keep only the most popular track
    - If popularity is fully missing → keep ALL tracks
    """

    def __init__(self, artist_col="artistID", popularity_col="popularity"):
        self.artist_col = artist_col
        self.popularity_col = popularity_col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Identify artists with at least one non-null popularity
        artist_has_popularity = X.groupby(self.artist_col)[self.popularity_col].apply(
            lambda x: x.notna().any()
        )

        artists_with_popularity = artist_has_popularity[artist_has_popularity].index
        artists_without_popularity = artist_has_popularity[~artist_has_popularity].index

        # For artists WITH popularity: keep only max popularity track
        mask_with_pop = X[self.artist_col].isin(artists_with_popularity)
        df_with_pop = X[mask_with_pop]

        if len(df_with_pop) > 0:
            # Get index of max popularity per artist
            idx_max = df_with_pop.groupby(self.artist_col)[self.popularity_col].idxmax()
            df_with_pop_filtered = X.loc[idx_max]
        else:
            df_with_pop_filtered = pd.DataFrame()

        # For artists WITHOUT popularity: keep all tracks
        mask_without_pop = X[self.artist_col].isin(artists_without_popularity)
        df_without_pop = X[mask_without_pop]

        # Combine and return
        result = pd.concat([df_with_pop_filtered, df_without_pop], ignore_index=True)

        return result
