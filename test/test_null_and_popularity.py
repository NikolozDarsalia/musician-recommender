import pandas as pd
from recommender_pipeline.preprocessors.artists_preprocessors.high_nulls_dropper import (
    HighNullDropper,
)
from recommender_pipeline.preprocessors.artists_preprocessors.popularity_imputer import (
    PopularityImputer,
)


def test_high_null_dropper_removes_columns():
    df = pd.DataFrame(
        {
            "a": [1, None, None, None],
            "b": [1, 2, 3, 4],
        }
    )

    dropper = HighNullDropper(threshold=0.6)
    out = dropper.fit_transform(df)

    assert "a" not in out.columns
    assert "b" in out.columns


def test_popularity_imputer_fills_nulls():
    df = pd.DataFrame(
        {
            "popularity": [50, None, 70],
            "num_listeners": [1000, 2000, 3000],
        }
    )

    imputer = PopularityImputer()
    out = imputer.fit_transform(df)

    assert out["popularity"].isnull().sum() == 0
