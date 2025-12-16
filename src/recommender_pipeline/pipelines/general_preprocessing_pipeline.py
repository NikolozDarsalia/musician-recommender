# global_preprocessing_pipeline.py
import pandas as pd


class GeneralPreprocessingPipeline:
    """
    Global preprocessing pipeline that enriches:
    - interactions  -> with USER-level aggregated tags
    - spotify       -> with ARTIST-level aggregated tags + listener counts
    """

    def __init__(self, how="left"):
        from recommender_pipeline.preprocessors.general_preprocessors.tag_aggregator import (
            TagAggregator,
        )
        from recommender_pipeline.preprocessors.general_preprocessors.tags_joiner import (
            JoinTags,
        )
        from recommender_pipeline.preprocessors.general_preprocessors.artist_listener_counter import (
            ArtistListenerCounter,
        )
        from recommender_pipeline.preprocessors.general_preprocessors.join_artist_listeners import (
            JoinArtistListeners,
        )

        # ----------------------------------
        # Tag aggregators
        # ----------------------------------

        # ARTIST-level tags (for spotify)
        self.artist_tag_aggregator = TagAggregator(
            group_by="artistID", tag_col="tagValue", output_col="artist_tags"
        )

        # ----------------------------------
        # Joiners
        # ----------------------------------
        self.join_artist_tags = JoinTags(id="artistID", how=how)

        self.listener_counter = ArtistListenerCounter()
        self.join_listeners = JoinArtistListeners()

    def run(
        self,
        interactions_df: pd.DataFrame,
        tag_df: pd.DataFrame,
        spotify_df: pd.DataFrame,
    ) -> dict:
        # ----------------------------------
        # 1. Aggregate ARTIST-level tags
        # ----------------------------------
        artist_tags = self.artist_tag_aggregator.transform(tag_df)

        # ----------------------------------
        # 3. Count artist listeners
        # ----------------------------------
        artist_listeners = self.listener_counter.transform(interactions_df)

        # ----------------------------------
        # 4. Enrich Spotify with ARTIST tags + listeners
        # ----------------------------------
        spotify_enriched = self.join_artist_tags.fit(spotify_df, artist_tags).transform(
            spotify_df
        )

        spotify_enriched = self.join_listeners.fit(
            spotify_enriched, artist_listeners
        ).transform(spotify_enriched)

        return {
            "spotify": spotify_enriched,
        }
