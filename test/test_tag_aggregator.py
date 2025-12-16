import pandas as pd
from recommender_pipeline.preprocessors.general_preprocessors.tag_aggregator import (
    TagAggregator,
)


def test_tag_aggregator_artist_level():
    df = pd.DataFrame(
        {
            "userID": [1, 2, 3],
            "artistID": ["a1", "a1", "a2"],
            "tagValue": ["rock", "indie", "jazz"],
        }
    )

    agg = TagAggregator(group_by="artistID", tag_col="tagValue")
    result = agg.transform(df)

    a1_tags = result.loc[result["artistID"] == "a1", "aggregated_tags"].iloc[0]

    assert "rock" in a1_tags
    assert "indie" in a1_tags
    assert "|" in a1_tags
    assert len(result) == 2


def test_tag_aggregator_deduplicates_tags():
    df = pd.DataFrame(
        {
            "artistID": ["a1", "a1"],
            "tagValue": ["rock", "rock"],
        }
    )

    agg = TagAggregator(group_by="artistID")
    result = agg.transform(df)

    assert result["aggregated_tags"].iloc[0] == "rock"
