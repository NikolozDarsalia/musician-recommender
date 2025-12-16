import pandas as pd
import numpy as np
import pytest
from recommender_pipeline.preprocessors.userwise_minmax_scaler import (
    UserMinMaxListeningScaler,
)


def test_basic_user_scaling():
    df = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2],
            "artistID": [10, 11, 12, 10, 13],
            "weight": [5, 10, 15, 3, 3],
        }
    )

    scaler = UserMinMaxListeningScaler()
    out = scaler.fit_transform(df)

    # User 1: min=5, max=15
    user1 = out[out.userID == 1].sort_values("artistID")
    assert np.allclose(user1.weight.values, [0.0, 0.5, 1.0])

    # User 2: single unique value -> filled with 1.0
    user2 = out[out.userID == 2]
    assert np.all(user2.weight.values == 1.0)


def test_output_columns_only():
    df = pd.DataFrame(
        {
            "userID": [1],
            "artistID": [100],
            "weight": [7],
            "extra_col": [999],
        }
    )

    scaler = UserMinMaxListeningScaler()
    out = scaler.fit_transform(df)

    assert list(out.columns) == ["userID", "artistID", "weight"]


def test_unseen_user_raises():
    train = pd.DataFrame(
        {
            "userID": [1, 1],
            "artistID": [10, 11],
            "weight": [2, 5],
        }
    )

    test = pd.DataFrame(
        {
            "userID": [2],
            "artistID": [12],
            "weight": [3],
        }
    )

    scaler = UserMinMaxListeningScaler()
    scaler.fit(train)

    with pytest.raises(ValueError):
        scaler.transform(test)


def test_single_interaction_user_fill_value():
    df = pd.DataFrame(
        {
            "userID": [1],
            "artistID": [10],
            "weight": [42],
        }
    )

    scaler = UserMinMaxListeningScaler(single_value_fill=0.5)
    out = scaler.fit_transform(df)

    assert out.weight.iloc[0] == 0.5
