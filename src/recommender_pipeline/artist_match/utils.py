import re
from typing import Any, Iterable, Optional, Tuple
from unidecode import unidecode
import pandas as pd
from rapidfuzz import process as _rf_process
from rapidfuzz import fuzz as _rf_fuzz


def _normalize(name: Any) -> str:
    """advanced normalization for artist names: lower, strip, remove punctuation and extra spaces."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower()
    s = unidecode(s)  # normalize unicode
    s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_simple(name: Any) -> str:
    """Simple normalization for artist names: lower, strip, remove punctuation and extra spaces."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _best_match(
    query: str, choices: Iterable[str], scorer=_rf_fuzz.ratio, score_cutoff: int = 0
) -> Optional[Tuple[str, int]]:
    res = _rf_process.extractOne(
        query, choices, scorer=scorer, score_cutoff=score_cutoff
    )
    # rapidfuzz returns (choice, score, index) or None
    if res:
        return res[0], int(res[1])
    return None
