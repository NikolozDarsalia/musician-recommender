import os
import joblib
import optuna
from lightfm import LightFM
from lightfm.evaluation import ndcg_at_k
from interfaces.base_tuner import BaseTuner


class LightFMTuner(BaseTuner):
    def __init__(
        self,
        n_trials=50,
        direction="maximize",
        k=10,
        param_space=None,
        num_threads=-1,
        model_path="lightfm_best_model.pkl",
    ):
        """
        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.
        direction : str
            "maximize" or "minimize".
        k : int
            Cutoff for NDCG@k evaluation.
        param_space : dict
            Dictionary defining parameter ranges for tuning.
        num_threads : int
            Number of threads for LightFM training (parallelism inside LightFM).
        model_path : str
            Path to save/load the best tuned model.
        """
        self.n_trials = n_trials
        self.direction = direction
        self.k = k
        self.param_space = param_space or {}
        self.num_threads = num_threads
        self.model_path = model_path
        self.best_params_ = None
        self.best_model_ = None

    def optimize(self, train, val, user_features=None, item_features=None):
        def objective(trial):
            # Suggest parameters dynamically from param_space
            no_components = trial.suggest_int(
                "no_components", *self.param_space.get("no_components", (20, 100))
            )
            learning_rate = trial.suggest_loguniform(
                "learning_rate", *self.param_space.get("learning_rate", (1e-4, 1e-1))
            )
            epochs = trial.suggest_int(
                "epochs", *self.param_space.get("epochs", (5, 30))
            )
            item_alpha = trial.suggest_loguniform(
                "item_alpha", *self.param_space.get("item_alpha", (1e-6, 1e-2))
            )
            user_alpha = trial.suggest_loguniform(
                "user_alpha", *self.param_space.get("user_alpha", (1e-6, 1e-2))
            )
            loss = trial.suggest_categorical(
                "loss", self.param_space.get("loss", ["warp", "bpr", "warp-kos"])
            )

            model = LightFM(
                no_components=no_components,
                learning_rate=learning_rate,
                loss=loss,
                item_alpha=item_alpha,
                user_alpha=user_alpha,
                random_state=42,
            )
            model.fit(
                train,
                user_features=user_features,
                item_features=item_features,
                epochs=epochs,
                num_threads=self.num_threads,
            )

            score = ndcg_at_k(
                model,
                val,
                user_features=user_features,
                item_features=item_features,
                k=self.k,
            ).mean()
            return score

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params_ = study.best_params
        self.best_model_ = LightFM(**self.best_params_)
        self.best_model_.fit(
            train,
            user_features=user_features,
            item_features=item_features,
            epochs=self.best_params_.get("epochs", 10),
            num_threads=self.num_threads,
        )

        # Save the best model immediately after training
        self.save()
        return self.best_model_

    def save(self):
        """
        Save the best tuned LightFM model and parameters to disk using joblib.
        """
        if self.best_model_ is None:
            raise RuntimeError("No model to save. Run optimize() first.")
        joblib.dump(
            {
                "model": self.best_model_,
                "best_params": self.best_params_,
                "n_trials": self.n_trials,
                "direction": self.direction,
                "k": self.k,
            },
            self.model_path,
        )

    def load(self):
        """
        Load the best tuned LightFM model and parameters from disk using joblib.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

        data = joblib.load(self.model_path)
        self.best_model_ = data["model"]
        self.best_params_ = data["best_params"]
        self.n_trials = data["n_trials"]
        self.direction = data["direction"]
        self.k = data["k"]
        return self.best_model_
