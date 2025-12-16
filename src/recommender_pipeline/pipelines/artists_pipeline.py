from ..preprocessors.artists_preprocessors.genre_tag_combiner import GenreTagCombiner
from ..preprocessors.artists_preprocessors.high_nulls_dropper import HighNullDropper
from ..preprocessors.artists_preprocessors.most_popular_track import (
    MostPopularTrackSelector,
)
from ..preprocessors.artists_preprocessors.numeric_feature_builder import (
    ArtistNumericFeatureBuilder,
)
from ..preprocessors.artists_preprocessors.popularity_export import PopularityExporter
from ..preprocessors.artists_preprocessors.popularity_imputer import PopularityImputer


from ..feature_generators.artists_features.genre_family_engineer import (
    HighLevelGenreFeatureGenerator,
)

from ..preprocessors.feature_scaler import LogMinMaxScaler

import pandas as pd


class ArtistMetadataPipeline:
    def __init__(self, numeric_cols):
        self.steps = [
            HighNullDropper(threshold=0.8),
            PopularityImputer(),
            GenreTagCombiner(),
            HighLevelGenreFeatureGenerator(),
            MostPopularTrackSelector(),
            PopularityExporter(),
            ArtistNumericFeatureBuilder(numeric_cols=numeric_cols),
            LogMinMaxScaler(),
        ]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            X = step.fit_transform(X)
            print(f"{step} is done!")
        return X
