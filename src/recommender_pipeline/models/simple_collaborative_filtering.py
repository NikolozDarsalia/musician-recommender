import os
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from interfaces.base_recommender import BaseRecommender


class CollaborativeFilteringRecommender(BaseRecommender):
    def __init__(self, n_components=50, model_path="cf_baseline.pkl"):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.user_factors = None
        self.item_factors = None
        self.model_path = model_path

    def fit(self, interactions_df, user_features=None, item_features=None):
        """
        interactions_df: sparse matrix [n_users x n_items]
        """
        self.user_factors = self.svd.fit_transform(interactions_df)
        self.item_factors = self.svd.components_.T
        return self

    def predict(self, user_ids, item_ids):
        return np.sum(self.user_factors[user_ids] * self.item_factors[item_ids], axis=1)

    def recommend(self, user_id, k=10):
        scores = self.user_factors[user_id] @ self.item_factors.T
        return np.argsort(-scores)[:k]

    def save(self):
        """
        Save the trained CF model and parameters to disk using joblib.
        """
        joblib.dump(
            {
                "svd": self.svd,
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
                "n_components": self.n_components,
            },
            self.model_path,
        )

    def load(self):
        """
        Load the CF model and parameters from disk using joblib.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

        data = joblib.load(self.model_path)
        self.svd = data["svd"]
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.n_components = data["n_components"]
        return self
