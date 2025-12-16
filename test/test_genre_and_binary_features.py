import pandas as pd
from recommender_pipeline.preprocessors.artists_preprocessors.genre_tag_combiner import (
    GenreTagCombiner,
)

from recommender_pipeline.feature_generators.artists_features.genre_family_engineer import (
    HighLevelGenreFeatureGenerator,
)


def test_genre_and_tag_combined():
    df = pd.DataFrame(
        {
            "track_genre": ["rock"],
            "aggregated_tags": ["british|indie"],
        }
    )

    combiner = GenreTagCombiner()
    out = combiner.transform(df)

    assert "genre_tag_text" in out.columns
    assert "rock" in out["genre_tag_text"].iloc[0]
    assert "british" in out["genre_tag_text"].iloc[0]


def test_binary_features_created():
    df = pd.DataFrame(
        {
            "tag_genre_text": ["british rock indie"],
        }
    )

    gen = HighLevelGenreFeatureGenerator()
    out = gen.transform(df)

    assert out["genre_rock"].iloc[0] == 1
    assert out["geo_british"].iloc[0] == 1
