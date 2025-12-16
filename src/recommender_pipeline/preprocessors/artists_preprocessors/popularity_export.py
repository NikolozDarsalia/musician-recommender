from ...interfaces.base_preprocessor import BasePreprocessor
import pandas as pd


class PopularityExporter(BasePreprocessor):
    def __init__(
        self,
        artist_col="artistID",
        popularity_col="popularity",
        output_path="artist_popularity.parquet",
    ):
        self.artist_col = artist_col
        self.popularity_col = popularity_col
        self.output_path = output_path

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pop_df = X.groupby(self.artist_col)[self.popularity_col].max().reset_index()
        pop_df.to_parquet(self.output_path, index=False)

        # Remove popularity from modeling dataset
        return X.drop(columns=[self.popularity_col])
