from typing import Tuple, Dict, Callable, Set
import pandas as pd
import numpy as np
from rapidfuzz import fuzz as _rf_fuzz
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from .utils import _normalize, _normalize_simple, _best_match


class ArtistMatcher:
    """
    Efficient artist name matching with blocking for large-scale fuzzy matching.
    """

    EXACT_MATCH_SCORE = 100
    DEFAULT_SCORE_CUTOFF = 85
    DEFAULT_SCORER = "ratio"
    DEFAULT_RIGHT_PREFIX = "right_"

    def __init__(
        self,
        right: pd.DataFrame,
        right_artist_col: str,
        score_cutoff: int = DEFAULT_SCORE_CUTOFF,
        scorer: str = DEFAULT_SCORER,
        use_blocking: bool = True,
        block_size: int = 3,
    ) -> None:
        """
        Initialize the matcher with blocking for performance.

        Args:
            right: Reference dataframe to match against
            right_artist_col: Column name containing artist names
            score_cutoff: Minimum fuzzy match score (0-100)
            scorer: RapidFuzz scorer name
            use_blocking: If True, use blocking keys to reduce comparisons
            block_size: Number of characters for blocking key (3-4 recommended)
        """
        if right_artist_col not in right.columns:
            raise KeyError(f"Column '{right_artist_col}' not found in right dataframe")

        if not 0 <= score_cutoff <= 100:
            raise ValueError(
                f"score_cutoff must be between 0 and 100, got {score_cutoff}"
            )

        try:
            scorer_func = getattr(_rf_fuzz, scorer)
        except AttributeError:
            raise AttributeError(f"Invalid scorer '{scorer}'")

        self.right = right.reset_index(drop=True)
        self.right_artist_col = right_artist_col
        self.score_cutoff = score_cutoff
        self.scorer = scorer_func
        self.use_blocking = use_blocking
        self.block_size = block_size

        print(f"Building lookup tables for {len(right)} artists...")
        self._build_lookup_tables()
        print("✓ Lookup tables ready")

    def _build_lookup_tables(self):
        """Build normalized name lookup tables with blocking."""
        self.right_names = (
            self.right[self.right_artist_col].astype(str).fillna("").tolist()
        )

        # Build normalized mappings
        self.normalized_right_map, self.canonical_name_for_norm = (
            self._build_normalized_maps(_normalize)
        )
        self.simple_normalized_right_map, self.simple_canonical_name_for_norm = (
            self._build_normalized_maps(_normalize_simple)
        )

        self.unique_right_normalized = list(self.normalized_right_map.keys())

        # Build blocking index for fuzzy matching
        if self.use_blocking:
            self._build_blocking_index()

    def _build_blocking_index(self):
        """Build blocking index to reduce fuzzy matching comparisons."""
        self.blocking_index: Dict[str, Set[str]] = defaultdict(set)

        for norm_name in self.unique_right_normalized:
            if len(norm_name) >= self.block_size:
                # Create multiple blocking keys per name for better recall
                blocks = [
                    norm_name[: self.block_size],  # Prefix
                    norm_name[-self.block_size :],  # Suffix
                ]

                # Add middle block if name is long enough
                if len(norm_name) >= self.block_size * 2:
                    mid = len(norm_name) // 2
                    blocks.append(norm_name[mid : mid + self.block_size])

                for block_key in blocks:
                    self.blocking_index[block_key].add(norm_name)
            else:
                # Short names: use entire name as block
                self.blocking_index[norm_name].add(norm_name)

        print(f"  Created {len(self.blocking_index)} blocking keys")

    def _get_blocking_candidates(self, query: str) -> Set[str]:
        """Get candidate matches using blocking."""
        if not self.use_blocking or len(query) < self.block_size:
            return set(self.unique_right_normalized)

        candidates = set()

        # Get candidates from multiple blocks
        blocks = [
            query[: self.block_size],
            query[-self.block_size :],
        ]

        if len(query) >= self.block_size * 2:
            mid = len(query) // 2
            blocks.append(query[mid : mid + self.block_size])

        for block_key in blocks:
            candidates.update(self.blocking_index.get(block_key, set()))

        # If no candidates, fall back to all (shouldn't happen often)
        if not candidates:
            return set(self.unique_right_normalized)

        return candidates

    def _build_normalized_maps(
        self, normalize_func: Callable
    ) -> Tuple[Dict[str, list], Dict[str, str]]:
        """Build normalized name mappings."""
        normalized_map: Dict[str, list] = {}
        for idx, name in enumerate(self.right_names):
            norm = normalize_func(name)
            normalized_map.setdefault(norm, []).append(idx)

        canonical_name_map = {
            norm: self.right.at[indices[0], self.right_artist_col]
            for norm, indices in normalized_map.items()
        }

        return normalized_map, canonical_name_map

    def _resolve_one_to_many(
        self, matches: Dict[int, Tuple[int, str, int]]
    ) -> Dict[int, Tuple[int, str, int]]:
        """Resolve one-to-many matches."""
        if not matches:
            return matches

        matches_df = pd.DataFrame(
            [
                {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "matched_name": matched_name,
                    "score": score,
                }
                for left_idx, (right_idx, matched_name, score) in matches.items()
            ]
        )

        best_matches = matches_df.loc[
            matches_df.groupby("matched_name")["score"].idxmax()
        ]

        return {
            int(row["left_idx"]): (
                int(row["right_idx"]),
                row["matched_name"],
                int(row["score"]),
            )
            for _, row in best_matches.iterrows()
        }

    def _build_result_dataframe(
        self,
        left: pd.DataFrame,
        matches: Dict[int, Tuple[int, str, int]],
        right_prefix: str,
        keep_unmatched: bool,
    ) -> pd.DataFrame:
        """Build result dataframe."""
        left_reset = left.reset_index(drop=True)

        match_score_col = f"{right_prefix}match_score"
        matched_artist_col = f"{right_prefix}matched_artist"

        if not matches:
            if keep_unmatched:
                result = left_reset.copy()
                for col in self.right.columns:
                    result[f"{right_prefix}{col}"] = pd.NA
                result[matched_artist_col] = pd.NA
                result[match_score_col] = pd.NA
                return result
            else:
                result = left_reset.iloc[0:0].copy()
                for col in self.right.columns:
                    result[f"{right_prefix}{col}"] = pd.NA
                result[matched_artist_col] = pd.NA
                result[match_score_col] = pd.NA
                return result

        result = left_reset.copy()

        for col in self.right.columns:
            result[f"{right_prefix}{col}"] = pd.NA
        result[matched_artist_col] = pd.NA
        result[match_score_col] = pd.NA

        matched_left_indices = []
        matched_right_indices = []
        matched_names = []
        matched_scores = []

        for left_idx, (right_idx, matched_name, score) in matches.items():
            matched_left_indices.append(left_idx)
            matched_right_indices.append(right_idx)
            matched_names.append(matched_name)
            matched_scores.append(score)

        result.loc[matched_left_indices, matched_artist_col] = matched_names
        result.loc[matched_left_indices, match_score_col] = matched_scores

        for col in self.right.columns:
            result.loc[matched_left_indices, f"{right_prefix}{col}"] = self.right.loc[
                matched_right_indices, col
            ].values

        if not keep_unmatched:
            result = result[result[match_score_col].notna()].reset_index(drop=True)

        return result

    def _fuzzy_match_batch(self, args):
        """Worker function for parallel fuzzy matching."""
        queries, candidates_list, scorer, score_cutoff = args
        results = []

        for query, candidates in zip(queries, candidates_list):
            match_res = _best_match(
                query, list(candidates), scorer=scorer, score_cutoff=score_cutoff
            )
            results.append(match_res)

        return results

    def match(
        self,
        left: pd.DataFrame,
        left_artist_col: str,
        right_prefix: str = DEFAULT_RIGHT_PREFIX,
        keep_unmatched: bool = True,
        one_to_many: bool = False,
    ) -> pd.DataFrame:
        """Match with blocking optimization."""
        if left_artist_col not in left.columns:
            raise KeyError(f"{left_artist_col} not found in left dataframe")

        print(f"\nMatching {len(left)} artists...")
        left_reset = left.reset_index(drop=True)

        # Vectorized normalization
        left_names = left_reset[left_artist_col].astype(str).fillna("")
        left_simple_norms = left_names.apply(_normalize_simple)

        matches = {}

        # Step 1: Exact matching
        print("  Step 1: Exact matching...")
        exact_mask = left_simple_norms.isin(self.simple_normalized_right_map.keys())
        exact_count = exact_mask.sum()
        exact_indices = np.where(exact_mask)[0]

        for idx in exact_indices:
            left_simple_norm = left_simple_norms.iloc[idx]
            right_idx = self.simple_normalized_right_map[left_simple_norm][0]
            matched_name = self.simple_canonical_name_for_norm[left_simple_norm]
            matches[idx] = (right_idx, matched_name, self.EXACT_MATCH_SCORE)

        print(f"    ✓ Found {exact_count} exact matches")

        # Step 2: Fuzzy matching with blocking
        unmatched_indices = np.where(~exact_mask)[0]

        # In match() method, replace fuzzy matching section:
        if len(unmatched_indices) > 0:
            print(
                f"  Step 2: Fuzzy matching {len(unmatched_indices)} unmatched artists (parallel)..."
            )
            left_norms = left_names.iloc[unmatched_indices].apply(_normalize).tolist()

            # Prepare candidates for each query
            candidates_list = [self._get_blocking_candidates(q) for q in left_norms]

            # Split into chunks for parallel processing
            n_workers = cpu_count()
            chunk_size = len(left_norms) // n_workers + 1

            tasks = []
            for i in range(0, len(left_norms), chunk_size):
                chunk_queries = left_norms[i : i + chunk_size]
                chunk_candidates = candidates_list[i : i + chunk_size]
                tasks.append(
                    (chunk_queries, chunk_candidates, self.scorer, self.score_cutoff)
                )

            # Process in parallel
            with Pool(n_workers) as pool:
                results_chunks = pool.map(self._fuzzy_match_batch, tasks)

            # Flatten results
            fuzzy_matches = 0
            results = [item for chunk in results_chunks for item in chunk]

            for i, match_res in enumerate(results):
                if match_res:
                    matched_norm, score = match_res
                    original_idx = unmatched_indices[i]
                    right_idx = self.normalized_right_map[matched_norm][0]
                    matched_name = self.canonical_name_for_norm[matched_norm]
                    matches[original_idx] = (right_idx, matched_name, int(score))
                    fuzzy_matches += 1

            print(f"    ✓ Found {fuzzy_matches} fuzzy matches")

        if not one_to_many:
            matches = self._resolve_one_to_many(matches)

        return self._build_result_dataframe(left, matches, right_prefix, keep_unmatched)
