from ..interfaces.base_feature_generator import BaseFeatureGenerator


class AudioFeatureInteractionEngineer(BaseFeatureGenerator):
    """
    Engineer interaction features from Spotify audio attributes.

    Parameters
    ----------
    feature_cols : list
        List of numeric Spotify audio feature columns to use.
    include_ratios : bool
        Whether to create ratio-type interaction features.
    include_products : bool
        Whether to create product (multiplicative) interaction features.
    include_composites : bool
        Whether to create handcrafted domain-specific composite features.
    eps : float
        Small constant to stabilize division.
    """

    def __init__(
        self,
        feature_cols=None,
        include_ratios=True,
        include_products=True,
        include_composites=True,
        eps=1e-6,
    ):
        if feature_cols is None:
            feature_cols = [
                "energy",
                "valence",
                "danceability",
                "loudness",
                "tempo",
                "acousticness",
                "instrumentalness",
            ]

        self.feature_cols = feature_cols
        self.include_ratios = include_ratios
        self.include_products = include_products
        self.include_composites = include_composites
        self.eps = eps
        self.generated_features_ = []

    def fit(self, X, y=None):
        # Nothing to calculate (no training) → return self
        return self

    def transform(self, X):
        X = X.copy()
        cols = self.feature_cols

        # Safety: keep only available columns
        cols = [c for c in cols if c in X.columns]

        # --------------------------
        # PRODUCT INTERACTIONS
        # --------------------------
        if self.include_products:
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    name = f"{c1}_x_{c2}"
                    X[name] = X[c1] * X[c2]
                    self.generated_features_.append(name)

        # --------------------------
        # RATIO INTERACTIONS
        # --------------------------
        if self.include_ratios:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j:
                        c1, c2 = cols[i], cols[j]
                        name = f"{c1}_div_{c2}"
                        X[name] = X[c1] / (X[c2] + self.eps)
                        self.generated_features_.append(name)

        # --------------------------
        # HANDCRAFTED COMPOSITES
        # --------------------------
        if self.include_composites:
            if "energy" in X and "loudness" in X:
                X["perceived_loudness_intensity"] = X["energy"] * X["loudness"]
                self.generated_features_.append("perceived_loudness_intensity")

            if "acousticness" in X and "valence" in X:
                X["mellowness_score"] = X["acousticness"] * X["valence"]
                self.generated_features_.append("mellowness_score")

            if "energy" in X and "acousticness" in X:
                X["aggressiveness_index"] = X["energy"] / (X["acousticness"] + self.eps)
                self.generated_features_.append("aggressiveness_index")

            if "valence" in X and "tempo" in X:
                X["cheerful_tempo_score"] = X["valence"] * X["tempo"]
                self.generated_features_.append("cheerful_tempo_score")

            if "danceability" in X and "energy" in X:
                X["dance_energy"] = X["danceability"] * X["energy"]
                self.generated_features_.append("dance_energy")

        return X
