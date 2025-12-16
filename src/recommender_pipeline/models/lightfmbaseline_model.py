import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from lightfm import LightFM

from ..interfaces.base_recommender import BaseRecommender


class LightFMBaselineModel(BaseRecommender):
    """
    Baseline LightFM model with training and evaluation capabilities.

    Features:
    - Trains LightFM with default/custom hyperparameters
    - Evaluates on multiple metrics
    - Supports item features (artist metadata)
    - Provides recommendations for users
    """

    def __init__(
        self,
        no_components: int = 64,
        loss: str = "warp",
        learning_rate: float = 0.05,
        epochs: int = 30,
        num_threads: int = 4,
        random_state: int = 42,
    ):
        """
        Initialize baseline model with hyperparameters.

        Args:
            no_components: Embedding dimensionality
            loss: Loss function ('warp', 'bpr', 'warp-kos')
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            num_threads: Number of threads for training
            random_state: Random seed for reproducibility
        """
        self.no_components = no_components
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_threads = num_threads
        self.random_state = random_state

        self.model = None
        self.item_features = None
        self.is_trained = False

    def fit(self, train_matrix, item_features=None, verbose=True):
        """
        Train the LightFM model.

        Args:
            train_matrix: Sparse interaction matrix (users x items)
            item_features: Optional sparse feature matrix (items x features)
            verbose: Whether to print training progress
        """
        self.model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        self.item_features = item_features

        self.model.fit(
            train_matrix,
            item_features=item_features,
            epochs=self.epochs,
            num_threads=self.num_threads,
            verbose=verbose,
        )

        self.is_trained = True
        return self

    def recommend(self, user_idx: int, k: int = 10, filter_items=None) -> np.ndarray:
        """
        Get top-K recommendations for a user.

        Args:
            user_idx: User index in the interaction matrix
            k: Number of recommendations
            filter_items: Optional set of item indices to exclude

        Returns:
            Array of top-K item indices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")

        # Get number of items from model
        n_items = self.model.item_embeddings.shape[0]

        # Predict scores for all items
        scores = self.model.predict(
            user_idx, np.arange(n_items), item_features=self.item_features
        )

        # Filter out items if specified
        if filter_items is not None:
            scores[list(filter_items)] = -np.inf

        # Get top-K
        top_items = np.argsort(-scores)[:k]
        return top_items

    def evaluate(self, test_matrix, metrics: List, k: int = 10) -> Dict[str, float]:
        """
        Evaluate model on test set with multiple metrics.

        Args:
            test_matrix: Sparse test interaction matrix
            metrics: List of metric objects (implementing BaseMetric interface)
            k: K value for ranking metrics

        Returns:
            Dictionary of metric names and values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        results = {}
        for metric in metrics:
            try:
                score = metric.compute(self, test_matrix, k=k)
                results[metric.name()] = score
            except Exception as e:
                print(f"Warning: Failed to compute {metric.name()}: {e}")
                results[metric.name()] = None

        return results

    def print_evaluation(
        self, train_matrix, val_matrix, test_matrix, metrics: List, k: int = 10
    ):
        """
        Evaluate and print metrics for train, validation, and test sets.

        Args:
            train_matrix: Training interaction matrix
            val_matrix: Validation interaction matrix
            test_matrix: Test interaction matrix
            metrics: List of metric objects
            k: K value for ranking metrics
        """
        print("=" * 70)
        print("BASELINE MODEL EVALUATION")
        print("=" * 70)

        print("\nModel Configuration:")
        print(f"  Loss: {self.loss}")
        print(f"  Embedding dim: {self.no_components}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {self.epochs}")

        # Evaluate on train set
        print(f"\n{'TRAIN SET':-^70}")
        train_results = self.evaluate(train_matrix, metrics, k=k)
        for metric_name, score in train_results.items():
            if score is not None:
                print(f"  {metric_name:20s}: {score:.6f}")

        # Evaluate on validation set
        print(f"\n{'VALIDATION SET':-^70}")
        val_results = self.evaluate(val_matrix, metrics, k=k)
        for metric_name, score in val_results.items():
            if score is not None:
                print(f"  {metric_name:20s}: {score:.6f}")

        # Evaluate on test set
        print(f"\n{'TEST SET':-^70}")
        test_results = self.evaluate(test_matrix, metrics, k=k)
        for metric_name, score in test_results.items():
            if score is not None:
                print(f"  {metric_name:20s}: {score:.6f}")

        print("=" * 70)
        print()

        return {"train": train_results, "val": val_results, "test": test_results}

    def save(self, path: str):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load model from disk"""
        with open(path, "rb") as f:
            return pickle.load(f)
