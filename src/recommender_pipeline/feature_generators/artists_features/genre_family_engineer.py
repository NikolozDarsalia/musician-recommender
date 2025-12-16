import pandas as pd
from ...interfaces.base_feature_generator import BaseFeatureGenerator


class HighLevelGenreFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, text_col="tag_genre_text"):
        self.text_col = text_col

        self.genre_map = {
            "rock": ["rock", "alt", "grunge", "punk"],
            "electronic": ["edm", "techno", "house", "trance"],
            "hip_hop": ["hip-hop", "rap"],
            "classical": ["classical", "opera"],
            "metal": ["metal", "hardcore"],
        }

        self.mood_map = {
            "happy": ["happy", "party"],
            "sad": ["sad", "melancholy"],
            "chill": ["chill", "ambient"],
            "energetic": ["energy", "dance"],
        }

        self.geo_map = {
            "british": ["british", "uk"],
            "latin": ["latin", "latino", "salsa"],
            "asian": ["j-pop", "k-pop", "mandopop"],
        }

    def _make_flags(self, X, mapping, prefix):
        for k, keywords in mapping.items():
            X[f"{prefix}_{k}"] = (
                X[self.text_col]
                .str.contains("|".join(keywords), case=False, na=False)
                .astype(int)
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        self._make_flags(X, self.genre_map, "genre")
        self._make_flags(X, self.mood_map, "mood")
        self._make_flags(X, self.geo_map, "geo")

        return X
