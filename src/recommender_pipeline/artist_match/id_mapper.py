from typing import Dict, Set, Optional, Tuple
import pandas as pd


class ArtistIDMapper:
    """
    Manages artist ID mappings between LastFM and Spotify data sources.

    This class handles the creation of a unified artist ID system by:
    1. Using LastFM artistID as the unified ID for matched artists
    2. Creating new unified IDs for unmatched Spotify artists
    3. Providing lookup functionality between different ID systems

    The mapper ensures that:
    - Matched artists use LastFM artistID as unified ID
    - Unmatched LastFM artists use their own artistID as unified ID
    - Unmatched Spotify artists get new unique unified IDs
    - All ID mappings are consistent and reversible
    """

    def __init__(self, start_id: Optional[int] = None):
        """
        Initialize the ID mapper.

        Args:
            start_id: Starting ID for new unified IDs for unmatched Spotify artists.
                     If None, will be set to max(lastfm_ids) + 1 during fit.
        """
        self.start_id = start_id
        self.next_available_id = start_id

        # Mapping dictionaries
        self.lastfm_to_unified: Dict[int, int] = {}
        self.spotify_to_unified: Dict[str, int] = {}
        self.unified_to_lastfm: Dict[int, int] = {}
        self.unified_to_spotify: Dict[int, str] = {}

        # Track which IDs are used
        self.used_unified_ids: Set[int] = set()

    def fit(
        self,
        matched_artists_df: pd.DataFrame,
        lastfm_artists_df: pd.DataFrame,
        spotify_artists_df: pd.DataFrame,
        lastfm_id_col: str = "artistID",
        spotify_id_col: str = "artist_name",
        matched_lastfm_col: str = "artistID",
        matched_spotify_col: str = "right_artist_name",
    ) -> "ArtistIDMapper":
        """
        Create unified ID mappings from matched and unmatched artist data.

        Args:
            matched_artists_df: DataFrame with matched LastFM-Spotify artist pairs
            lastfm_artists_df: DataFrame with all LastFM artists
            spotify_artists_df: DataFrame with all Spotify artists
            lastfm_id_col: Column name for LastFM artist ID
            spotify_id_col: Column name for Spotify artist name
            matched_lastfm_col: Column name for LastFM ID in matched data
            matched_spotify_col: Column name for Spotify name in matched data

        Returns:
            self for method chaining
        """
        # Reset mappings
        self._reset_mappings()

        # Determine start_id if not provided
        if self.start_id is None:
            max_lastfm_id = lastfm_artists_df[lastfm_id_col].max()
            self.start_id = int(max_lastfm_id) + 1
            self.next_available_id = self.start_id

        # Step 1: Process matched artists - they use LastFM ID as unified ID
        self._process_matched_artists(
            matched_artists_df, matched_lastfm_col, matched_spotify_col
        )

        # Step 2: Process unmatched LastFM artists - they use their own ID as unified ID
        self._process_unmatched_lastfm(lastfm_artists_df, lastfm_id_col)

        # Step 3: Process unmatched Spotify artists - they get new IDs
        self._process_unmatched_spotify(spotify_artists_df, spotify_id_col)

        return self

    def _reset_mappings(self):
        """Reset all internal mappings."""
        if self.start_id is not None:
            self.next_available_id = self.start_id
        self.lastfm_to_unified.clear()
        self.spotify_to_unified.clear()
        self.unified_to_lastfm.clear()
        self.unified_to_spotify.clear()
        self.used_unified_ids.clear()

    def _process_matched_artists(
        self, matched_df: pd.DataFrame, lastfm_col: str, spotify_col: str
    ):
        """Process matched artists - VECTORIZED VERSION."""
        if len(matched_df) == 0:
            return

        # Vectorized operations
        lastfm_ids = matched_df[lastfm_col].astype(int).values
        spotify_ids = matched_df[spotify_col].astype(str).values

        # Batch update dictionaries
        for lastfm_id, spotify_id in zip(lastfm_ids, spotify_ids):
            unified_id = lastfm_id

            self.lastfm_to_unified[lastfm_id] = unified_id
            self.spotify_to_unified[spotify_id] = unified_id
            self.unified_to_lastfm[unified_id] = lastfm_id
            self.unified_to_spotify[unified_id] = spotify_id
            self.used_unified_ids.add(unified_id)

    def _process_unmatched_lastfm(self, lastfm_df: pd.DataFrame, id_col: str):
        """Process unmatched LastFM artists - VECTORIZED VERSION."""
        if len(lastfm_df) == 0:
            return

        # Get unmatched IDs
        lastfm_ids = lastfm_df[id_col].astype(int)
        unmatched_mask = ~lastfm_ids.isin(self.lastfm_to_unified.keys())
        unmatched_ids = lastfm_ids[unmatched_mask].values

        # Batch update
        for lastfm_id in unmatched_ids:
            unified_id = lastfm_id

            self.lastfm_to_unified[lastfm_id] = unified_id
            self.unified_to_lastfm[unified_id] = lastfm_id
            self.used_unified_ids.add(unified_id)

    def _process_unmatched_spotify(self, spotify_df: pd.DataFrame, id_col: str):
        """Process unmatched Spotify artists - VECTORIZED VERSION."""
        if len(spotify_df) == 0:
            return

        # Get unmatched IDs
        spotify_ids = spotify_df[id_col].astype(str)
        unmatched_mask = ~spotify_ids.isin(self.spotify_to_unified.keys())
        unmatched_ids = spotify_ids[unmatched_mask].values

        # Batch update
        for spotify_id in unmatched_ids:
            unified_id = self._get_next_available_id()

            self.spotify_to_unified[spotify_id] = unified_id
            self.unified_to_spotify[unified_id] = spotify_id

    def _get_next_available_id(self) -> int:
        """Get the next available unified ID."""
        current_id = self.next_available_id
        self.used_unified_ids.add(current_id)
        self.next_available_id += 1
        return current_id

    def get_unified_id(self, source_id, source: str = "auto") -> Optional[int]:
        """
        Get unified ID for a given source ID.

        Args:
            source_id: The original ID from LastFM or Spotify
            source: 'lastfm', 'spotify', or 'auto' to detect

        Returns:
            Unified ID or None if not found
        """
        if source == "auto":
            # Try both mappings
            if isinstance(source_id, int) and source_id in self.lastfm_to_unified:
                return self.lastfm_to_unified[source_id]
            elif str(source_id) in self.spotify_to_unified:
                return self.spotify_to_unified[str(source_id)]
            return None
        elif source == "lastfm":
            return self.lastfm_to_unified.get(source_id)
        elif source == "spotify":
            return self.spotify_to_unified.get(str(source_id))
        else:
            raise ValueError(
                f"Invalid source: {source}. Must be 'lastfm', 'spotify', or 'auto'"
            )

    def get_source_id(self, unified_id: int, target_source: str) -> Optional:
        """
        Get original source ID for a unified ID.

        Args:
            unified_id: The unified ID
            target_source: 'lastfm' or 'spotify'

        Returns:
            Original source ID or None if not found
        """
        if target_source == "lastfm":
            return self.unified_to_lastfm.get(unified_id)
        elif target_source == "spotify":
            return self.unified_to_spotify.get(unified_id)
        else:
            raise ValueError(
                f"Invalid target_source: {target_source}. Must be 'lastfm' or 'spotify'"
            )

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        id_col: str,
        source: str,
        unified_id_col: str = "unified_artist_id",
    ) -> pd.DataFrame:
        """
        Add unified IDs to a dataframe.

        Args:
            df: Input dataframe
            id_col: Column containing source IDs
            source: 'lastfm' or 'spotify'
            unified_id_col: Name for new unified ID column

        Returns:
            DataFrame with added unified ID column
        """
        result_df = df.copy()
        result_df[unified_id_col] = result_df[id_col].apply(
            lambda x: self.get_unified_id(x, source)
        )
        return result_df

    def get_mapping_summary(self) -> Dict:
        """
        Get summary statistics about the mappings.

        Returns:
            Dictionary with mapping statistics
        """
        return {
            "total_unified_ids": len(self.used_unified_ids),
            "lastfm_artists": len(self.lastfm_to_unified),
            "spotify_artists": len(self.spotify_to_unified),
            "matched_artists": len(
                set(self.unified_to_lastfm.keys()) & set(self.unified_to_spotify.keys())
            ),
            "unmatched_lastfm": len(self.unified_to_lastfm)
            - len(
                set(self.unified_to_lastfm.keys()) & set(self.unified_to_spotify.keys())
            ),
            "unmatched_spotify": len(self.unified_to_spotify)
            - len(
                set(self.unified_to_lastfm.keys()) & set(self.unified_to_spotify.keys())
            ),
            "id_range": (min(self.used_unified_ids), max(self.used_unified_ids))
            if self.used_unified_ids
            else (None, None),
        }

    def export_mappings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export mapping tables as DataFrames for final parquet files.

        Returns:
            Tuple of (lastfm_mappings_df, spotify_mappings_df)
        """
        lastfm_df = pd.DataFrame(
            [
                {"lastfm_artist_id": lastfm_id, "unified_artist_id": unified_id}
                for lastfm_id, unified_id in self.lastfm_to_unified.items()
            ]
        )

        spotify_df = pd.DataFrame(
            [
                {"spotify_artist_name": spotify_id, "unified_artist_id": unified_id}
                for spotify_id, unified_id in self.spotify_to_unified.items()
            ]
        )

        return lastfm_df, spotify_df
