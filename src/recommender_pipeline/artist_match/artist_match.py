from typing import Optional, Tuple, Dict, Callable
import pandas as pd
from rapidfuzz import fuzz as _rf_fuzz

from .utils import _normalize, _normalize_simple, _best_match


class ArtistMatcher:
    """
    Efficient artist name matching using exact and fuzzy matching strategies.
    
    Encapsulates the matching logic and lookup tables for reusable artist matching
    between dataframes with configurable scoring and normalization.
    """
    
    # Constants
    EXACT_MATCH_SCORE = 100
    DEFAULT_SCORE_CUTOFF = 85
    DEFAULT_SCORER = "ratio"
    DEFAULT_RIGHT_PREFIX = "right_"
    
    def __init__(self, 
                 right: pd.DataFrame, 
                 right_artist_col: str,
                 score_cutoff: int = DEFAULT_SCORE_CUTOFF,
                 scorer: str = DEFAULT_SCORER) -> None:
        """
        Initialize the matcher with the reference dataset.
        
        Args:
            right: Reference dataframe to match against
            right_artist_col: Column name containing artist names in right dataframe
            score_cutoff: Minimum fuzzy match score (0-100) to accept
            scorer: RapidFuzz scorer name to use for fuzzy matching
            
        Raises:
            KeyError: If right_artist_col is not found in the dataframe
            ValueError: If score_cutoff is not between 0 and 100
            AttributeError: If scorer is not a valid RapidFuzz scorer
        """
        if right_artist_col not in right.columns:
            raise KeyError(f"Column '{right_artist_col}' not found in right dataframe")
        
        if not 0 <= score_cutoff <= 100:
            raise ValueError(f"score_cutoff must be between 0 and 100, got {score_cutoff}")
        
        try:
            scorer_func = getattr(_rf_fuzz, scorer)
        except AttributeError:
            raise AttributeError(f"Invalid scorer '{scorer}'. Must be a valid RapidFuzz scorer function")
        
        self.right = right.reset_index(drop=True)
        self.right_artist_col = right_artist_col
        self.score_cutoff = score_cutoff
        self.scorer = scorer_func
        
        # Build lookup tables once during initialization
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build normalized name lookup tables for efficient matching."""
        self.right_names = self.right[self.right_artist_col].astype(str).fillna("").tolist()
        
        # Build normalized mappings
        self.normalized_right_map, self.canonical_name_for_norm = self._build_normalized_maps(_normalize)
        self.simple_normalized_right_map, self.simple_canonical_name_for_norm = self._build_normalized_maps(_normalize_simple)
        
        # Unique normalized names for fuzzy matching
        self.unique_right_normalized = list(self.normalized_right_map.keys())
    
    def _build_normalized_maps(self, normalize_func: Callable) -> Tuple[Dict[str, list], Dict[str, str]]:
        """Build normalized name mappings for efficient lookups.
        
        Args:
            normalize_func: Function to normalize artist names
            
        Returns:
            Tuple of (normalized_map, canonical_name_map)
        """
        normalized_map: Dict[str, list] = {}  # normalized_name -> list of indices
        for idx, name in enumerate(self.right_names):
            norm = normalize_func(name)
            normalized_map.setdefault(norm, []).append(idx)
        
        # Create canonical name mapping (normalized -> original name from first occurrence)
        canonical_name_map = {norm: self.right.at[indices[0], self.right_artist_col] 
                             for norm, indices in normalized_map.items()}
        
        return normalized_map, canonical_name_map
    
    def find_match_for_artist(self, left_name: str) -> Optional[Tuple[int, str, int]]:
        """
        Find the best match for a single artist name using exact then fuzzy matching.
        
        Returns:
            Tuple of (right_idx, matched_name, score) if match found, None otherwise
        """
        left_simple_norm = _normalize_simple(left_name)
        
        # First try exact match with simple normalization
        if left_simple_norm in self.simple_normalized_right_map:
            right_idx = self.simple_normalized_right_map[left_simple_norm][0]
            matched_name = self.simple_canonical_name_for_norm[left_simple_norm]
            return (right_idx, matched_name, self.EXACT_MATCH_SCORE)
        
        # Try fuzzy matching with advanced normalization
        left_norm = _normalize(left_name)
        match_res = _best_match(left_norm, self.unique_right_normalized, 
                               scorer=self.scorer, score_cutoff=self.score_cutoff)
        if match_res:
            matched_norm, score = match_res
            right_idx = self.normalized_right_map[matched_norm][0]
            matched_name = self.canonical_name_for_norm[matched_norm]
            return (right_idx, matched_name, int(score))
        
        return None

    def _resolve_one_to_many(self, matches: Dict[int, Tuple[int, str, int]]) -> Dict[int, Tuple[int, str, int]]:
        """
        Resolve one-to-many matches by keeping only the highest scoring match for each right artist.
        
        When multiple left artists match to the same right artist, this method keeps only
        the match with the highest score to ensure one-to-one relationships.
        
        Args:
            matches: Dictionary mapping left_idx -> (right_idx, matched_name, score)
            
        Returns:
            Filtered matches dictionary with only best matches per right artist
        """
        if not matches:
            return matches
            
        # Group matches by matched_name and keep only the highest scoring match for each
        best_match_per_artist = {}  # matched_name -> (left_idx, score)
        for left_idx, (right_idx, matched_name, score) in matches.items():
            current_best = best_match_per_artist.get(matched_name)
            if current_best is None or score > current_best[1]:
                best_match_per_artist[matched_name] = (left_idx, score)
        
        # Keep only the best matches
        best_left_indices = {left_idx for left_idx, _ in best_match_per_artist.values()}
        return {left_idx: match_info for left_idx, match_info in matches.items() 
                if left_idx in best_left_indices}

    def _create_matches_dataframe(self, 
                                  matches: Dict[int, Tuple[int, str, int]],
                                  right_prefix: str) -> pd.DataFrame:
        """Create a dataframe from the matches dictionary."""
        match_score_col = f"{right_prefix}match_score"
        matched_artist_col = f"{right_prefix}matched_artist"
        
        match_data = []
        for left_idx, (right_idx, matched_name, score) in matches.items():
            match_data.append({
                '_left_idx': left_idx,
                '_right_idx': right_idx,
                matched_artist_col: matched_name,
                match_score_col: score
            })
        return pd.DataFrame(match_data)

    def _prepare_right_for_merge(self, right_prefix: str) -> pd.DataFrame:
        """Prepare the right dataframe for merging with proper prefixes."""
        right_reset = self.right.copy()
        right_reset['_right_idx'] = right_reset.index
        
        # Add prefix to right columns before merge
        right_cols_rename = {col: f"{right_prefix}{col}" for col in self.right.columns}
        return right_reset.rename(columns=right_cols_rename)

    def _add_empty_columns(self, 
                          result: pd.DataFrame, 
                          right_prefix: str) -> pd.DataFrame:
        """Add empty right columns when no matches are found."""
        match_score_col = f"{right_prefix}match_score"
        matched_artist_col = f"{right_prefix}matched_artist"
        
        for col in self.right.columns:
            result[f"{right_prefix}{col}"] = pd.NA
        result[matched_artist_col] = pd.NA
        result[match_score_col] = pd.NA
        return result

    def _finalize_result_columns(self, 
                                result: pd.DataFrame, 
                                left: pd.DataFrame,
                                right_prefix: str) -> pd.DataFrame:
        """Clean up temporary columns and ensure proper ordering."""
        match_score_col = f"{right_prefix}match_score"
        matched_artist_col = f"{right_prefix}matched_artist"
        
        # Clean up temporary index columns
        result = result.drop(columns=['_left_idx'], errors='ignore')
        result = result.drop(columns=['_right_idx'], errors='ignore')
        
        # Ensure proper column order
        final_columns = list(left.columns) + [f"{right_prefix}{c}" for c in self.right.columns] + [matched_artist_col, match_score_col]
        return result.reindex(columns=final_columns)

    def _build_result_dataframe(self, 
                                left: pd.DataFrame,
                                matches: Dict[int, Tuple[int, str, int]],
                                right_prefix: str,
                                keep_unmatched: bool) -> pd.DataFrame:
        """
        Build the final result dataframe by merging left data with matched right data.
        
        Args:
            left: Original left dataframe
            matches: Dictionary mapping left_idx -> (right_idx, matched_name, score)
            right_prefix: Prefix for right dataframe columns
            keep_unmatched: Whether to keep unmatched left rows
            
        Returns:
            Final merged dataframe with proper column ordering
        """
        left_reset = left.reset_index(drop=True)
        left_reset['_left_idx'] = left_reset.index
        
        if matches:
            # Create and merge matches dataframe
            matches_df = self._create_matches_dataframe(matches, right_prefix)
            how = 'left' if keep_unmatched else 'inner'
            result = left_reset.merge(matches_df, on='_left_idx', how=how)
            
            # Merge with right dataframe
            right_prefixed = self._prepare_right_for_merge(right_prefix)
            result = result.merge(right_prefixed, on='_right_idx', how='left')
            
        else:
            # No matches found
            result = left_reset.copy()
            if keep_unmatched:
                result = self._add_empty_columns(result, right_prefix)
            else:
                # Return empty result with proper columns
                result = result.iloc[0:0].copy()
                result = self._add_empty_columns(result, right_prefix)
        
        return self._finalize_result_columns(result, left, right_prefix)

    def match(self,
              left: pd.DataFrame,
              left_artist_col: str,
              right_prefix: str = DEFAULT_RIGHT_PREFIX,
              keep_unmatched: bool = True,
              one_to_many: bool = False) -> pd.DataFrame:
        """
        Match left dataframe against the reference dataframe using two-step process:
        1. First perform exact matching using simple normalization
        2. Then perform fuzzy matching on remaining unmatched items

        Args:
            left: left dataframe (will be returned with matched columns from right).
            left_artist_col: column name in left containing artist name.
            right_prefix: prefix to add to columns brought from the right dataframe.
            keep_unmatched: if True, keep left rows that have no match (matched columns will be NaN).
            one_to_many: if False, each right artist can only match to one left artist (highest score wins).

        Returns:
            A dataframe containing all columns from left plus matched columns from right prefixed with right_prefix,
            and two extra columns: '{right_prefix}matched_artist' and '{right_prefix}match_score'.
            Exact matches get a score of 100, fuzzy matches get their computed score.
        """
        if left_artist_col not in left.columns:
            raise KeyError(f"{left_artist_col} not found in left dataframe")

        # Step 1: Perform all matching (exact then fuzzy) without merging
        matches = {}  # left_idx -> (right_idx, matched_name, score)
        
        for idx, left_row in left.reset_index(drop=True).iterrows():
            left_name = left_row.get(left_artist_col, "")
            match_result = self.find_match_for_artist(left_name)
            
            if match_result:
                matches[idx] = match_result

        # Step 2: Handle one-to-many logic on matches
        if not one_to_many:
            matches = self._resolve_one_to_many(matches)

        # Step 3: Build result using pandas merge operations
        return self._build_result_dataframe(left, matches, right_prefix, keep_unmatched)