import pandas as pd
import pytest
from recommender_pipeline.preprocessors.artists_preprocessors.numeric_feature_builder import (
    ArtistNumericFeatureBuilder,
)


def test_numeric_mean_aggregation_with_global_fallback():
    """Test numeric columns are averaged per artist, nulls filled with global mean."""
    df = pd.DataFrame(
        {
            "artistID": ["a", "a", "b", "b"],
            "energy": [
                0.8,
                0.6,
                None,
                None,
            ],  # Artist a: 0.7, Artist b: global mean 0.7
            "genre_rock": [1, 1, 0, 0],
        }
    )

    builder = ArtistNumericFeatureBuilder(numeric_cols=["energy"])
    out = builder.transform(df)

    # Check no nulls remain
    assert out["energy"].isnull().sum() == 0

    # Check artist-level aggregation
    assert len(out) == 2  # Two artists
    assert out.index.name == "artistID"
    assert out.loc["a", "energy"] == 0.7  # mean(0.8, 0.6)
    assert out.loc["b", "energy"] == 0.7  # global mean


def test_binary_mode_aggregation():
    """Test binary columns take mode (most common value) per artist."""
    df = pd.DataFrame(
        {
            "artistID": ["x", "x", "x", "y", "y"],
            "popularity": [80, 90, 85, 50, 60],
            "genre_rock": [1, 1, 0, 0, 1],  # x: mode=1, y: mode=0 or 1 (tie)
            "mood_happy": [1, 1, 1, 0, 0],  # x: mode=1, y: mode=0
        }
    )

    builder = ArtistNumericFeatureBuilder(numeric_cols=["popularity"])
    out = builder.transform(df)

    # Check binary columns exist and are integers
    assert "genre_rock" in out.columns
    assert "mood_happy" in out.columns
    assert out["genre_rock"].dtype == int
    assert out["mood_happy"].dtype == int

    # Check modes
    assert out.loc["x", "genre_rock"] == 1  # 1 appears twice, 0 once
    assert out.loc["x", "mood_happy"] == 1  # 1 appears three times
    assert out.loc["y", "mood_happy"] == 0  # 0 appears twice


def test_artist_level_output_with_index():
    """Test output is artist-level with artistID as index."""
    df = pd.DataFrame(
        {
            "artistID": [1, 1, 1, 2, 2, 3],
            "track_id": [101, 102, 103, 201, 202, 301],
            "danceability": [0.8, 0.7, 0.9, 0.5, 0.6, 0.4],
            "genre_pop": [1, 1, 0, 1, 1, 0],
        }
    )

    builder = ArtistNumericFeatureBuilder(numeric_cols=["danceability"])
    out = builder.transform(df)

    # Check artist-level output
    assert len(out) == 3  # Three unique artists
    assert out.index.name == "artistID"
    assert list(out.index) == [1, 2, 3]

    # Check no track_id in output
    assert "track_id" not in out.columns

    # Check aggregated values
    assert out.loc[1, "danceability"] == pytest.approx(0.8)  # mean(0.8, 0.7, 0.9)
    assert out.loc[2, "danceability"] == pytest.approx(0.55)  # mean(0.5, 0.6)


def test_all_null_numeric_uses_global_mean():
    """Test artist with all null numeric values gets global mean."""
    df = pd.DataFrame(
        {
            "artistID": ["m", "m", "n", "n"],
            "popularity": [80, 90, None, None],  # m: 85, n: global mean 85
            "genre_rock": [1, 0, 1, 1],
        }
    )

    builder = ArtistNumericFeatureBuilder(numeric_cols=["popularity"])
    out = builder.transform(df)

    assert out.loc["m", "popularity"] == 85.0
    assert out.loc["n", "popularity"] == 85.0  # Global mean
