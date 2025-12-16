from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseTuner(ABC):
    """
    Interface for hyperparameter optimization tuners.
    Defines the contract that all tuner implementations must follow.
    """

    @abstractmethod
    def objective(self, trial: Any, metrics: List) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optimization trial object (e.g., Optuna Trial)
            metrics: List of metric objects for evaluation

        Returns:
            Optimization metric score
        """
        pass

    @abstractmethod
    def optimize(
        self,
        metrics: List,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> Any:
        """
        Run hyperparameter optimization.

        Args:
            metrics: List of metric objects
            n_trials: Number of optimization trials
            timeout: Time limit in seconds (optional)
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress bar

        Returns:
            Best trained model
        """
        pass

    @abstractmethod
    def save_best_model(self, path: str):
        """
        Save the best model to disk.

        Args:
            path: File path to save the model
        """
        pass

    @abstractmethod
    def plot_optimization_history(self):
        """
        Plot optimization history using the optimizer's visualization tools.
        """
        pass
