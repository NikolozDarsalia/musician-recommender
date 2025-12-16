import pandas as pd
from recommender_pipeline.preprocessors.artists_preprocessors.most_popular_track import (
    MostPopularTrackSelector,
)


def test_most_popular_track_selector_mixed_null_and_non_null():
    """Test that artists with popularity keep only max, artists without keep all."""
    df = pd.DataFrame(
        {
            "artistID": ["a", "a", "b", "b"],
            "track_id": [1, 2, 3, 4],
            "popularity": [10, 50, None, None],
        }
    )

    selector = MostPopularTrackSelector()
    out = selector.transform(df)

    # Artist "a" → only most popular track kept
    a_tracks = out[out["artistID"] == "a"].reset_index(drop=True)
    assert len(a_tracks) == 1, f"Expected 1 track for artist 'a', got {len(a_tracks)}"
    assert a_tracks["track_id"].iloc[0] == 2, (
        f"Expected track_id=2, got {a_tracks['track_id'].iloc[0]}"
    )
    assert a_tracks["popularity"].iloc[0] == 50, "Expected popularity=50"

    # Artist "b" → all tracks kept (popularity fully null)
    b_tracks = out[out["artistID"] == "b"].reset_index(drop=True)
    assert len(b_tracks) == 2, f"Expected 2 tracks for artist 'b', got {len(b_tracks)}"
    assert set(b_tracks["track_id"]) == {3, 4}, (
        f"Expected track_ids {{3, 4}}, got {set(b_tracks['track_id'])}"
    )
    assert b_tracks["popularity"].isna().all(), (
        "Expected all popularity to be null for artist 'b'"
    )

    # Overall size check
    assert len(out) == 3, f"Expected 3 total rows, got {len(out)}"

    print("✓ Test passed!")


def test_most_popular_track_selector_all_have_popularity():
    """Test that when all artists have popularity, only max is kept per artist."""
    df = pd.DataFrame(
        {
            "artistID": ["x", "x", "x", "y", "y"],
            "track_id": [1, 2, 3, 4, 5],
            "popularity": [30, 80, 50, 90, 20],
        }
    )

    selector = MostPopularTrackSelector()
    out = selector.transform(df)

    # Should only have 2 rows (1 per artist)
    assert len(out) == 2, f"Expected 2 rows, got {len(out)}"

    # Artist "x" should have track_id=2 (popularity=80)
    x_tracks = out[out["artistID"] == "x"]
    assert len(x_tracks) == 1
    assert x_tracks["track_id"].iloc[0] == 2
    assert x_tracks["popularity"].iloc[0] == 80

    # Artist "y" should have track_id=4 (popularity=90)
    y_tracks = out[out["artistID"] == "y"]
    assert len(y_tracks) == 1
    assert y_tracks["track_id"].iloc[0] == 4
    assert y_tracks["popularity"].iloc[0] == 90

    print("✓ Test passed!")


def test_most_popular_track_selector_all_null_popularity():
    """Test that when all artists have null popularity, all tracks are kept."""
    df = pd.DataFrame(
        {
            "artistID": ["m", "m", "n", "n", "n"],
            "track_id": [1, 2, 3, 4, 5],
            "popularity": [None, None, None, None, None],
        }
    )

    selector = MostPopularTrackSelector()
    out = selector.transform(df)

    # All tracks should be kept
    assert len(out) == 5, f"Expected 5 rows, got {len(out)}"
    assert set(out["track_id"]) == {1, 2, 3, 4, 5}

    print("✓ Test passed!")


def test_most_popular_track_selector_partial_null_per_artist():
    """Test artist with SOME null popularities - should keep only max non-null."""
    df = pd.DataFrame(
        {
            "artistID": ["z", "z", "z", "z"],
            "track_id": [1, 2, 3, 4],
            "popularity": [None, 60, None, 40],
        }
    )

    selector = MostPopularTrackSelector()
    out = selector.transform(df)

    # Should keep only track_id=2 (max non-null popularity)
    assert len(out) == 1, f"Expected 1 row, got {len(out)}"
    assert out["track_id"].iloc[0] == 2
    assert out["popularity"].iloc[0] == 60

    print("✓ Test passed!")


def test_most_popular_track_selector_empty_dataframe():
    """Test with empty dataframe."""
    df = pd.DataFrame(
        {
            "artistID": [],
            "track_id": [],
            "popularity": [],
        }
    )

    selector = MostPopularTrackSelector()
    out = selector.transform(df)

    assert len(out) == 0, f"Expected 0 rows, got {len(out)}"
    assert list(out.columns) == ["artistID", "track_id", "popularity"]

    print("✓ Test passed!")


def test_most_popular_track_selector_single_track_per_artist():
    """Test when each artist has only 1 track."""
    df = pd.DataFrame(
        {
            "artistID": ["p", "q", "r"],
            "track_id": [1, 2, 3],
            "popularity": [50, None, 70],
        }
    )

    selector = MostPopularTrackSelector()
    out = selector.transform(df)

    # All tracks should be kept (only 1 per artist)
    assert len(out) == 3, f"Expected 3 rows, got {len(out)}"
    assert set(out["track_id"]) == {1, 2, 3}

    print("✓ Test passed!")


# Run all tests
if __name__ == "__main__":
    test_most_popular_track_selector_mixed_null_and_non_null()
    test_most_popular_track_selector_all_have_popularity()
    test_most_popular_track_selector_all_null_popularity()
    test_most_popular_track_selector_partial_null_per_artist()
    test_most_popular_track_selector_empty_dataframe()
    test_most_popular_track_selector_single_track_per_artist()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
