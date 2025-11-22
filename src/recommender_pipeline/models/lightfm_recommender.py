import os
import joblib
import numpy as np
from interfaces.base_recommender import BaseRecommender
from lightfm import LightFM


class LightFMRecommender(BaseRecommender):
    def __init__(
        self,
        no_components=30,
        loss="warp",
        epochs=20,
        num_threads=-1,
        model_path="lightfm_basemodel.pkl",
    ):
        self.model = LightFM(no_components=no_components, loss=loss)
        self.no_components = no_components
        self.loss = loss
        self.epochs = epochs
        self.num_threads = num_threads
        self.model_path = model_path

    def fit(self, interactions, user_features=None, item_features=None):
        self.model.fit(
            interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=self.epochs,
            num_threads=self.num_threads,
        )
        return self

    def predict(self, user_ids, item_ids):
        return self.model.predict(user_ids, item_ids)

    def recommend(self, user_id, k=10):
        n_items = self.model.item_embeddings.shape[0]
        all_items = np.arange(n_items)
        scores = self.model.predict(user_id, all_items)
        top_k = np.argsort(-scores)[:k]
        return top_k

    def save(self):
        """
        Save the trained LightFM model and parameters to disk using joblib.
        """
        joblib.dump(
            {
                "model": self.model,
                "no_components": self.no_components,
                "loss": self.loss,
                "epochs": self.epochs,
                "num_threads": self.num_threads,
            },
            self.model_path,
        )

    def load(self):
        """
        Load the LightFM model and parameters from disk using joblib.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.no_components = data["no_components"]
        self.loss = data["loss"]
        self.epochs = data["epochs"]
        self.num_threads = data["num_threads"]
        return self
