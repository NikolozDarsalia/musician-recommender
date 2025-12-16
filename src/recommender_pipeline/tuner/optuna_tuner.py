import optuna
from ..interfaces.base_tuner import BaseTuner
import pickle
from pathlib import Path
from typing import List, Optional
from optuna.trial import Trial

from recommender_pipeline.models.lightfmbaseline_model import LightFMBaselineModel


class LightFMOptunaOptimizer(BaseTuner):
    """
    Hyperparameter optimization for LightFM using Optuna.

    Features:
    - Bayesian optimization of LightFM hyperparameters
    - Supports multiple optimization metrics
    - Early stopping for efficiency
    - Saves best model automatically
    """

    def __init__(
        self,
        train_matrix,
        val_matrix,
        item_features=None,
        matrix_builder=None,
        features_builder=None,
        optimization_metric="ndcg@10",
        direction="maximize",
        k: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize optimizer.

        Args:
            train_matrix: Training interaction matrix
            val_matrix: Validation interaction matrix
            item_features: Optional item feature matrix
            matrix_builder: InteractionMatrixBuilder instance (for saving)
            features_builder: ArtistFeaturesBuilder instance (for saving)
            optimization_metric: Metric to optimize (must match metric.name())
            direction: 'maximize' or 'minimize'
            k: K value for ranking metrics
            random_state: Random seed
        """
        self.train_matrix = train_matrix
        self.val_matrix = val_matrix
        self.item_features = item_features
        self.matrix_builder = matrix_builder
        self.features_builder = features_builder
        self.optimization_metric = optimization_metric
        self.direction = direction
        self.k = k
        self.random_state = random_state

        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.study = None

    def objective(self, trial: Trial, metrics: List) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            metrics: List of metric objects for evaluation

        Returns:
            Optimization metric score
        """
        # Define hyperparameter search space
        params = {
            "no_components": trial.suggest_int("no_components", 16, 128, step=16),
            "loss": trial.suggest_categorical("loss", ["warp", "bpr", "warp-kos"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "epochs": trial.suggest_int("epochs", 10, 50, step=5),
            "random_state": self.random_state,
        }

        # Train model with suggested hyperparameters
        model = LightFMBaselineModel(**params)
        model.fit(self.train_matrix, item_features=self.item_features, verbose=False)

        # Evaluate on validation set
        val_results = model.evaluate(self.val_matrix, metrics, k=self.k)

        # Get optimization metric score
        score = val_results.get(self.optimization_metric)

        if score is None:
            raise ValueError(f"Metric {self.optimization_metric} not found in results")

        # Log all metrics to trial
        for metric_name, metric_value in val_results.items():
            if metric_value is not None:
                trial.set_user_attr(metric_name, metric_value)

        return score

    def optimize(
        self,
        metrics: List,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> LightFMBaselineModel:
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
        print("=" * 70)
        print("STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("=" * 70)
        print(f"Optimization metric: {self.optimization_metric}")
        print(f"Direction: {self.direction}")
        print(f"Number of trials: {n_trials}")
        print(f"K value: {self.k}")
        print()

        # Create Optuna study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Run optimization
        self.study.optimize(
            lambda trial: self.objective(trial, metrics),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        # Get best parameters and score
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"\nBest {self.optimization_metric}: {self.best_score:.6f}")
        print("\nBest hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param:20s}: {value}")

        # Train final model with best parameters
        print("\n" + "-" * 70)
        print("Training final model with best hyperparameters...")
        print("-" * 70)

        self.best_model = LightFMBaselineModel(**self.best_params)
        self.best_model.fit(
            self.train_matrix, item_features=self.item_features, verbose=True
        )

        print("\nFinal model trained successfully!")
        print("=" * 70)
        print()

        return self.best_model

    def save_pipeline_components(self, models_dir="models", test_matrix=None):
        """
        Save all pipeline components needed for predictions.

        Args:
            models_dir: Directory to save components
            test_matrix: Optional test matrix to save alongside train/val
        """
        print("\n" + "=" * 70)
        print("SAVING PIPELINE COMPONENTS")
        print("=" * 70)

        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)

        # 1. Save interaction matrices
        matrices_to_save = {"train": self.train_matrix, "val": self.val_matrix}
        if test_matrix is not None:
            matrices_to_save["test"] = test_matrix

        matrices_path = models_path / "interaction_matrices.pkl"
        with open(matrices_path, "wb") as f:
            pickle.dump(matrices_to_save, f)
        print(f"✓ Interaction matrices saved to: {matrices_path}")

        # 2. Save matrix builder (contains user/artist mappings)
        if self.matrix_builder is not None:
            builder_path = models_path / "matrix_builder.pkl"
            with open(builder_path, "wb") as f:
                pickle.dump(self.matrix_builder, f)
            print(f"✓ Matrix builder saved to: {builder_path}")
        else:
            print("⚠ Matrix builder not provided, skipping...")

        # 3. Save artist features matrix
        if self.item_features is not None:
            features_path = models_path / "artist_features.pkl"
            with open(features_path, "wb") as f:
                pickle.dump(self.item_features, f)
            print(f"✓ Artist features saved to: {features_path}")
        else:
            print("⚠ Artist features not provided, skipping...")

        # 4. Save features builder
        if self.features_builder is not None:
            features_builder_path = models_path / "features_builder.pkl"
            with open(features_builder_path, "wb") as f:
                pickle.dump(self.features_builder, f)
            print(f"✓ Features builder saved to: {features_builder_path}")
        else:
            print("⚠ Features builder not provided, skipping...")

        print("\nAll available components saved successfully!")
        print("=" * 70)

    def save_best_model(
        self,
        path: str,
        save_components: bool = True,
        models_dir: str = "models",
        test_matrix=None,
    ):
        """
        Save the best model to disk.

        Args:
            path: Path to save the best model
            save_components: Whether to also save pipeline components
            models_dir: Directory for pipeline components
            test_matrix: Optional test matrix to save
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run optimize() first.")

        # Save the model
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.best_model.save(path)
        print(f"Best model saved to: {path}")

        # Also save optimization results
        results_path = Path(path).parent / "optimization_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(
                {
                    "best_params": self.best_params,
                    "best_score": self.best_score,
                    "study": self.study,
                },
                f,
            )
        print(f"Optimization results saved to: {results_path}")

        # Save pipeline components if requested
        if save_components:
            self.save_pipeline_components(
                models_dir=models_dir, test_matrix=test_matrix
            )

    def plot_optimization_history(self):
        """Plot optimization history using Optuna's built-in visualization"""
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
            )

            fig1 = plot_optimization_history(self.study)
            fig1.show()

            fig2 = plot_param_importances(self.study)
            fig2.show()
        except ImportError:
            print("Install plotly for visualization: pip install plotly")
