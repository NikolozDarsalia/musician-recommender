import pandas as pd
from recommender_pipeline.artist_match.artist_match import ArtistMatcher
from recommender_pipeline.artist_match.id_mapper import ArtistIDMapper


def generate_unified_artist_mappings(
    lastfm_artists_df: pd.DataFrame,
    spotify_tracks_df: pd.DataFrame,
    lastfm_artist_name_col: str = "name",
    lastfm_artist_id_col: str = "artistID",
    spotify_artist_name_col: str = "artist_name",
    output_lastfm_path: str = "../data/lastfm_unified_artists.parquet",
    output_spotify_path: str = "../data/spotify_unified_artists.parquet",
    score_cutoff: int = 85,
    scorer: str = "ratio",
):
    """
    Generate two parquet files with unified artist IDs linking LastFM and Spotify data.

    NOTE: spotify_tracks_df is TRACK-LEVEL data (can have duplicate artists).
          Output will be track-level with unified_artist_id added.

    Args:
        lastfm_artists_df: DataFrame with LastFM artists (artist-level)
        spotify_tracks_df: DataFrame with Spotify tracks (track-level, has duplicate artists)
        lastfm_artist_name_col: Column name for artist name in LastFM data
        lastfm_artist_id_col: Column name for artistID in LastFM data
        spotify_artist_name_col: Column name for artist name in Spotify data
        output_lastfm_path: Path for output LastFM parquet file
        output_spotify_path: Path for output Spotify parquet file
        score_cutoff: Minimum fuzzy match score (0-100)
        scorer: RapidFuzz scorer name

    Returns:
        Tuple of (lastfm_output_df, spotify_output_df)
    """

    # Step 1: Extract unique Spotify artists from track-level data
    print("Step 1: Extracting unique Spotify artists from track data...")
    spotify_unique_artists = (
        spotify_tracks_df[[spotify_artist_name_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"  Total Spotify tracks: {len(spotify_tracks_df)}")
    print(f"  Unique Spotify artists: {len(spotify_unique_artists)}")
    print(f"  LastFM artists: {len(lastfm_artists_df)}")

    # Step 2: Match unique Spotify artists to LastFM artists
    print("\nStep 2: Matching Spotify artists to LastFM artists...")
    matcher = ArtistMatcher(
        right=lastfm_artists_df,
        right_artist_col=lastfm_artist_name_col,
        score_cutoff=score_cutoff,
        scorer=scorer,
        use_blocking=True,
        block_size=3,
    )

    matched_df = matcher.match(
        left=spotify_unique_artists,
        left_artist_col=spotify_artist_name_col,
        right_prefix="right_",
        keep_unmatched=True,
        one_to_many=False,
    )

    print("\n  Matching results:")
    print(f"    Matched artists: {matched_df['right_match_score'].notna().sum()}")
    print(
        f"    Unmatched Spotify artists: {matched_df['right_match_score'].isna().sum()}"
    )

    # Step 3: Create matched artists dataframe
    print("\nStep 3: Creating matched artists dataframe...")

    # Check what columns we actually have
    print(f"  Available columns in matched_df: {matched_df.columns.tolist()}")

    # Find the right column name (it should be right_{lastfm_artist_name_col} or similar)
    right_id_col = f"right_{lastfm_artist_id_col}"

    # The matched_df should have the original spotify column and right_ prefixed lastfm columns
    matched_artists = matched_df[matched_df["right_match_score"].notna()].copy()

    if right_id_col not in matched_artists.columns:
        raise KeyError(
            f"Column '{right_id_col}' not found. "
            f"Available columns: {matched_artists.columns.tolist()}"
        )

    matched_artists_final = matched_artists[
        [right_id_col, spotify_artist_name_col]
    ].copy()
    matched_artists_final.columns = [lastfm_artist_id_col, spotify_artist_name_col]

    print(f"  Matched pairs: {len(matched_artists_final)}")

    # Step 4: Create ID mapper
    print("\nStep 4: Creating unified ID mappings...")
    id_mapper = ArtistIDMapper()

    id_mapper.fit(
        matched_artists_df=matched_artists_final,
        lastfm_artists_df=lastfm_artists_df,
        spotify_artists_df=spotify_unique_artists,
        lastfm_id_col=lastfm_artist_id_col,
        spotify_id_col=spotify_artist_name_col,
        matched_lastfm_col=lastfm_artist_id_col,
        matched_spotify_col=spotify_artist_name_col,
    )

    # Print mapping summary
    summary = id_mapper.get_mapping_summary()
    print("\n  Mapping Summary:")
    print(f"    Total unified IDs: {summary['total_unified_ids']}")
    print(f"    LastFM artists: {summary['lastfm_artists']}")
    print(f"    Spotify artists: {summary['spotify_artists']}")
    print(f"    Matched artists: {summary['matched_artists']}")
    print(f"    Unmatched LastFM: {summary['unmatched_lastfm']}")
    print(f"    Unmatched Spotify: {summary['unmatched_spotify']}")
    print(f"    ID range: {summary['id_range']}")

    # Step 5: Export mapping tables
    print("\nStep 5: Generating output dataframes...")
    lastfm_mappings, spotify_mappings = id_mapper.export_mappings()

    # Step 6: Create outputs
    # LastFM output: lastfm_artist_name, lastfm_artist_id, unified_artist_id
    lastfm_output = lastfm_artists_df[
        [lastfm_artist_name_col, lastfm_artist_id_col]
    ].merge(
        lastfm_mappings,
        left_on=lastfm_artist_id_col,
        right_on="lastfm_artist_id",
        how="left",
    )
    lastfm_output = lastfm_output[
        [lastfm_artist_name_col, lastfm_artist_id_col, "unified_artist_id"]
    ]
    lastfm_output.columns = [
        "lastfm_artist_name",
        "lastfm_artist_id",
        "unified_artist_id",
    ]

    # Spotify output: TRACK-LEVEL with unified_artist_id added
    # Merge the track-level data with artist mappings
    spotify_output = spotify_tracks_df.merge(
        spotify_mappings,
        left_on=spotify_artist_name_col,
        right_on="spotify_artist_name",
        how="left",
    )

    # Keep all original columns plus unified_artist_id
    # Drop the duplicate spotify_artist_name column from the merge
    if (
        "spotify_artist_name" in spotify_output.columns
        and spotify_artist_name_col != "spotify_artist_name"
    ):
        spotify_output = spotify_output.drop(columns=["spotify_artist_name"])

    print(f"  LastFM output shape: {lastfm_output.shape}")
    print(f"  Spotify output shape: {spotify_output.shape} (track-level)")
    print(
        f"  Spotify tracks with unified ID: {spotify_output['unified_artist_id'].notna().sum()}"
    )
    print(
        f"  Spotify tracks without unified ID: {spotify_output['unified_artist_id'].isna().sum()}"
    )

    # Step 7: Save to parquet
    print("\nStep 6: Saving to parquet files...")
    lastfm_output.to_parquet(output_lastfm_path, index=False)
    print(f"  ✓ Saved: {output_lastfm_path}")

    spotify_output.to_parquet(output_spotify_path, index=False)
    print(f"  ✓ Saved: {output_spotify_path}")

    print("\n✓ Complete!")
    print("\nOutput structure:")
    print(f"  1. {output_lastfm_path}:")
    print("     - lastfm_artist_name, lastfm_artist_id, unified_artist_id")
    print(f"     - {len(lastfm_output)} rows (artist-level)")
    print(f"\n  2. {output_spotify_path}:")
    print("     - All original Spotify columns + unified_artist_id")
    print(f"     - {len(spotify_output)} rows (track-level, same as input)")
    print("     - Artists can appear multiple times (one per track)")
    print("     - unified_artist_id = NULL for unmatched artists")

    return lastfm_output, spotify_output
