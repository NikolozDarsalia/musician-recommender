from ...interfaces.base_preprocessor import BasePreprocessor
import pandas as pd


class GenreTagCombiner(BasePreprocessor):
    def __init__(self, genre_col="track_genre", tag_col="artist_tags"):
        self.genre_col = genre_col
        self.tag_col = tag_col
        self.out_col = "tag_genre_text"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[self.genre_col] = X[self.genre_col].fillna("")
        X[self.tag_col] = X[self.tag_col].fillna("")

        X[self.out_col] = (
            X[self.genre_col].astype(str) + "|" + X[self.tag_col].astype(str)
        ).str.strip("|")

        return X
