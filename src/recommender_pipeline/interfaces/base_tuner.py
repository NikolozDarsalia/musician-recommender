from abc import ABC, abstractmethod


class BaseTuner(ABC):
    @abstractmethod
    def optimize(self, train, val, user_features, item_features, n_trials, direction):
        """
        train: training interactions
        val: validation interactions
        user_features, item_features: sparse matrices
        n_trials: number of Optuna trials
        direction: maximize or minimize
        """
        raise NotImplementedError
