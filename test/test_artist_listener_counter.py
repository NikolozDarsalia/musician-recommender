import pandas as pd
from recommender_pipeline.preprocessors.general_preprocessors.artist_listener_counter import (
    ArtistListenerCounter,
)


def test_artist_listener_counter_counts_unique_users():
    interactions = pd.DataFrame(
        {
            "userID": [1, 1, 2, 3],
            "artistID": ["a1", "a1", "a1", "a2"],
        }
    )

    counter = ArtistListenerCounter()
    result = counter.transform(interactions)

    a1_listeners = result.loc[result["artistID"] == "a1", "num_listeners"].iloc[0]
    a2_listeners = result.loc[result["artistID"] == "a2", "num_listeners"].iloc[0]

    assert a1_listeners == 2
    assert a2_listeners == 1
