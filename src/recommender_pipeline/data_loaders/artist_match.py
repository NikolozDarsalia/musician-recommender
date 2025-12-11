from typing import Optional, Iterable, Tuple, Dict, Any
import re
from unidecode import unidecode

import pandas as pd
from rapidfuzz import process as _rf_process
from rapidfuzz import fuzz as _rf_fuzz

def _best_match(query: str, 
                choices: Iterable[str], 
                scorer=_rf_fuzz.ratio, 
                score_cutoff: int = 0) -> Optional[Tuple[str, int]]:
    res = _rf_process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)
    # rapidfuzz returns (choice, score, index) or None
    if res:
        return res[0], int(res[1])
    return None

def _normalize(name: Any) -> str:
    """Simple normalization for artist names: lower, strip, remove punctuation and extra spaces."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower()
    s = unidecode(s)  # normalize unicode
    s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_data_by_artist(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_artist_col: str,
    right_artist_col: str,
    score_cutoff: int = 85,
    right_prefix: str = "right_",
    keep_unmatched: bool = True,
    scorer: Optional[str] = "ratio",
) -> pd.DataFrame:
    """
    Fuzzy-match two dataframes on artist name columns.

    Args:
      left: left dataframe (will be returned with matched columns from right).
      right: right dataframe to match against.
      left_artist_col: column name in left containing artist name.
      right_artist_col: column name in right containing artist name.
      score_cutoff: minimum match score (0-100) to consider a match. Lower to be more permissive.
      right_prefix: prefix to add to columns brought from the right dataframe.
      keep_unmatched: if True, keep left rows that have no match (matched columns will be NaN).
      scorer: name of scorer to use with rapidfuzz (if available). Defaults to 'token_sort_ratio'.

    Returns:
      A dataframe containing all columns from left plus matched columns from right prefixed with right_prefix,
      and two extra columns: '{right_prefix}matched_artist' and '{right_prefix}match_score'.
    """
    if left_artist_col not in left.columns:
        raise KeyError(f"{left_artist_col} not found in left dataframe")
    if right_artist_col not in right.columns:
        raise KeyError(f"{right_artist_col} not found in right dataframe")

    # prepare choices from right
    right = right.reset_index(drop=True)
    right_names = right[right_artist_col].astype(str).fillna("").tolist()
    # map normalized name -> list of indices that have that normalized representation
    normalized_right_map: Dict[str, list] = {}
    for idx, name in enumerate(right_names):
        norm = _normalize(name)
        normalized_right_map.setdefault(norm, []).append(idx)

    # for matching use unique normalized right names and keep original representative name
    unique_right_normalized = list(normalized_right_map.keys())
    # also keep a mapping from normalized to a canonical display name from the right DF
    canonical_name_for_norm = {norm: right.at[indices[0], right_artist_col] for norm, indices in normalized_right_map.items()}

    rf_scorer = getattr(_rf_fuzz, scorer)

    # build result rows
    rows = []
    for idx, left_row in left.reset_index(drop=True).iterrows():
        left_name = left_row.get(left_artist_col, "")
        left_norm = _normalize(left_name)
        match_res = _best_match(left_norm, unique_right_normalized, scorer=rf_scorer, score_cutoff=score_cutoff)
        if match_res:
            matched_norm, score = match_res
            # pick first right row that corresponds to this normalized name
            right_idx = normalized_right_map[matched_norm][0]
            right_row = right.loc[right_idx]
            # prepare merged row: left fields, then right fields prefixed, plus matched name & score
            merged = left_row.to_dict()
            for c, v in right_row.items():
                merged[f"{right_prefix}{c}"] = v
            merged[f"{right_prefix}matched_artist"] = canonical_name_for_norm[matched_norm]
            merged[f"{right_prefix}match_score"] = int(score)
            rows.append(merged)
        else:
            if keep_unmatched:
                merged = left_row.to_dict()
                # add right columns as NaN
                for c in right.columns:
                    merged[f"{right_prefix}{c}"] = pd.NA
                merged[f"{right_prefix}matched_artist"] = pd.NA
                merged[f"{right_prefix}match_score"] = pd.NA
                rows.append(merged)
            else:
                # skip unmatched row
                continue

    result = pd.DataFrame(rows, columns=list(left.columns) + [f"{right_prefix}{c}" for c in right.columns] + [f"{right_prefix}matched_artist", f"{right_prefix}match_score"])
    return result