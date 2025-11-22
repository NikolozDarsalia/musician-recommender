from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class BaseTuner(ABC, BaseEstimator):
    """
    Abstract base class for hyperparameter tuners compatible with scikit-learn pipelines.
    Designed for recommenders (e.g., LightFM).
    """

    @abstractmethod
    def optimize(self, train, val, user_features=None, item_features=None):
        """
        Run hyperparameter optimization (e.g., with Optuna).
        Must set self.best_params_ and self.best_model_.
        """
        raise NotImplementedError

    def fit(self, train, y=None, user_features=None, item_features=None):
        """
        Fit by running optimization and storing the best model.
        """
        self.optimize(
            train, val=None, user_features=user_features, item_features=item_features
        )
        return self

    def predict(self, user_ids, item_ids):
        """
        Delegate prediction to the tuned model.
        """
        if self.best_model_ is None:
            raise RuntimeError("Model not optimized yet. Call fit() first.")
        return self.best_model_.predict(user_ids, item_ids)
