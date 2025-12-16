import pandas as pd
from recommender_pipeline.pipelines.artists_pipeline import ArtistMetadataPipeline


def test_artist_pipeline_runs_end_to_end():
    df = pd.DataFrame(
        {
            "artistID": [1, 2, 3],
            "track_genre": ["rock", None, "electronic"],
            "artist_tags": ["british", "edm", "energy"],
            "popularity": [50, None, 80],
            "num_listeners": [1000, 2000, 3000],
            "energy": [0.8, 0.7, 0.9],
            "danceability": [0.6, 0.5, 0.8],
        }
    )

    pipeline = ArtistMetadataPipeline(numeric_cols=["energy", "danceability"])

    out = pipeline.fit_transform(df)

    assert out.shape[0] == 3
    assert not out.isnull().any().any()
