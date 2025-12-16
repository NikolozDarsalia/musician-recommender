import pandas as pd
import numpy as np
from recommender_pipeline.preprocessors.feature_scaler import LogMinMaxScaler


def test_continuous_features_are_scaled():
    """Test continuous features get log + minmax scaling."""
    df = pd.DataFrame(
        {
            "artistID": [1, 2, 3],
            "popularity": [10, 50, 100],  # Continuous (3 unique)
            "danceability": [0.2, 0.5, 0.9],  # Continuous (3 unique)
            "genre_rock": [0, 1, 0],  # Binary (2 unique)
        }
    )

    scaler = LogMinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Check continuous features are scaled to [0, 1]
    assert scaled["popularity"].min() >= 0
    assert scaled["popularity"].max() <= 1
    assert scaled["danceability"].min() >= 0
    assert scaled["danceability"].max() <= 1

    # Check values are different from original (scaled)
    assert not np.allclose(scaled["popularity"].values, df["popularity"].values)

    # Check binary feature unchanged
    assert scaled["genre_rock"].tolist() == [0, 1, 0]

    # Check ID column unchanged
    assert scaled["artistID"].tolist() == [1, 2, 3]


def test_binary_features_remain_unchanged():
    """Test binary features (≤2 unique values) are not scaled."""
    df = pd.DataFrame(
        {
            "artistID": [1, 2, 3, 4],
            "energy": [0.3, 0.6, 0.9, 0.4],  # Continuous
            "is_major": [0, 1, 1, 0],  # Binary
            "has_vocals": [1, 1, 0, 1],  # Binary
        }
    )

    scaler = LogMinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Binary features should be identical
    assert scaled["is_major"].tolist() == df["is_major"].tolist()
    assert scaled["has_vocals"].tolist() == df["has_vocals"].tolist()

    # Continuous feature should be scaled
    assert not np.allclose(scaled["energy"].values, df["energy"].values)


def test_explicit_feature_list():
    """Test explicit feature_cols parameter."""
    df = pd.DataFrame(
        {
            "artistID": [1, 2, 3],
            "popularity": [10, 50, 100],
            "danceability": [0.2, 0.5, 0.9],
            "energy": [0.3, 0.6, 0.8],
        }
    )

    # Only scale popularity and energy
    scaler = LogMinMaxScaler(feature_cols=["popularity", "energy"])
    scaled = scaler.fit_transform(df)
    print(scaled)
    # Danceability should be unchanged
    assert scaled["danceability"].tolist() == df["danceability"].tolist()


def test_id_columns_excluded():
    """Test ID columns are automatically excluded from scaling."""
    df = pd.DataFrame(
        {
            "userID": [100, 200, 300],
            "artistID": [1, 2, 3],
            "unified_artist_id": [10, 20, 30],
            "popularity": [10, 50, 100],
        }
    )

    scaler = LogMinMaxScaler()
    scaled = scaler.fit_transform(df)

    # ID columns should remain unchanged
    assert scaled["userID"].tolist() == [100, 200, 300]
    assert scaled["artistID"].tolist() == [1, 2, 3]
    assert scaled["unified_artist_id"].tolist() == [10, 20, 30]


def test_transform_on_new_data():
    """Test fitted scaler works on new data."""
    train_df = pd.DataFrame(
        {
            "artistID": [1, 2, 3],
            "popularity": [10, 50, 100],
        }
    )

    test_df = pd.DataFrame(
        {
            "artistID": [4, 5],
            "popularity": [30, 70],
        }
    )

    scaler = LogMinMaxScaler()
    scaler.fit(train_df)

    # Transform new data
    scaled_test = scaler.transform(test_df)

    # Check scaled values are in [0, 1] range
    assert scaled_test["popularity"].min() >= 0
    assert scaled_test["popularity"].max() <= 1

    # Check artistID unchanged
    assert scaled_test["artistID"].tolist() == [4, 5]


def test_handles_zero_values_with_offset():
    """Test log transformation handles zeros correctly."""
    df = pd.DataFrame(
        {
            "artistID": [1, 2, 3, 4],
            "popularity": [0, 10, 50, 100],  # Contains zero
        }
    )

    scaler = LogMinMaxScaler(log_offset=1e-10)
    scaled = scaler.fit_transform(df)

    # Should not have NaN or inf values
    assert not scaled["popularity"].isna().any()
    assert not np.isinf(scaled["popularity"]).any()

    # Should be scaled to [0, 1]
    assert scaled["popularity"].min() >= 0
    assert scaled["popularity"].max() <= 1
