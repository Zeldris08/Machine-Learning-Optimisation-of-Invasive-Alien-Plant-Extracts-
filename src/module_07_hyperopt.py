"""
Module 7 — Hyperparameter Optimisation
Uses Optuna (Bayesian optimisation) to tune Random Forest and XGBoost
hyperparameters via CV on the training set only.
The test set is NEVER referenced during optimisation.

Usage:
    python src/module_07_hyperopt.py [--debug]
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, load_json, save_json, set_seeds

ENDPOINTS = ["germination_percent", "root_length_mm", "shoot_length_mm"]


def make_rf_objective(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, seed: int):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 500]),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, None]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": seed,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=kf,
                                  scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(-scores.mean())

    return objective


def make_xgb_objective(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, seed: int):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 500]),
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4, 6]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 0.3]),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.1, 1.0]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [0.1, 1.0, 10.0]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5]),
            "random_state": seed,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda" 
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=kf,
                                  scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(-scores.mean())

    return objective


def optimise_endpoint(
    endpoint: str,
    feat_df: pd.DataFrame,
    targets: pd.DataFrame,
    split_indices: dict,
    config: dict,
    logger,
) -> dict:
    if endpoint not in targets.columns:
        logger.info(f"Endpoint '{endpoint}' not in dataset; skipping.")
        return {}

    y = targets[endpoint].dropna()
    X = feat_df.loc[y.index].copy()

    train_idx = split_indices[endpoint]["train_indices"]
    train_idx = [i for i in train_idx if i in X.index]
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    seed = config["RANDOM_SEED"]
    n_trials = config["MODELS"]["n_search_iterations"]
    cv_folds = config["MODELS"]["cv_folds"]

    ensure_dirs("outputs/hyperparameter_search")
    results = {}

    for model_name, objective_fn in [
        ("random_forest", make_rf_objective(X_train, y_train, cv_folds, seed)),
        ("xgboost", make_xgb_objective(X_train, y_train, cv_folds, seed)),
    ]:
        logger.info(f"Optimising {model_name} for '{endpoint}' ({n_trials} trials)...")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"  Best CV RMSE ({model_name}): {best_value:.4f}")
        logger.info(f"  Best params: {best_params}")

        # Save trial history
        trials_df = study.trials_dataframe()
        trials_df.to_csv(
            f"outputs/hyperparameter_search/{model_name}_{endpoint}_trials.csv",
            index=False,
        )

        # Retrain on full training set with best params
        if model_name == "random_forest":
            best_model = RandomForestRegressor(
                **best_params, random_state=seed, n_jobs=-1
            )
        else:
            best_model = xgb.XGBRegressor(
                **best_params, random_state=seed, verbosity=0, n_jobs=-1
            )

        best_model.fit(X_train, y_train)
        model_path = f"models/{model_name}_{endpoint}_tuned.pkl"
        joblib.dump(best_model, model_path)

        results[model_name] = {
            "best_cv_rmse": best_value,
            "best_params": best_params,
            "model_path": model_path,
        }

    # Select best between RF and XGBoost (same logic as Module 6)
    rf_rmse = results["random_forest"]["best_cv_rmse"]
    xgb_rmse = results["xgboost"]["best_cv_rmse"]
    if xgb_rmse < rf_rmse and (rf_rmse - xgb_rmse) / rf_rmse > 0.05:
        best_name = "xgboost"
    else:
        best_name = "random_forest"

    results["selected_model"] = best_name
    best_model_path = results[best_name]["model_path"]

    # Copy as canonical best_model for this endpoint
    import shutil
    shutil.copy(best_model_path, f"models/best_model_{endpoint}.pkl")
    logger.info(f"Best tuned model for '{endpoint}': {best_name} → models/best_model_{endpoint}.pkl")

    return results


def run(config: dict, debug: bool = False) -> dict:
    ensure_dirs("models", "outputs/hyperparameter_search")

    with ModuleRunner(
        "module_07_hyperopt",
        config,
        input_files=["data/features/osmotic_corrected_matrix.csv"],
        output_files=["models/hyperparams.json"],
        debug=debug,
    ) as logger:
        seed = config["RANDOM_SEED"]
        set_seeds(seed)

        feat_df = pd.read_csv("data/features/osmotic_corrected_matrix.csv")
        full_df = pd.read_csv("data/features/full_matrix_with_targets.csv")
        split_indices = load_json("models/split_indices.json")

        target_cols = [c for c in ENDPOINTS if c in full_df.columns]
        targets = full_df[target_cols]

        all_results = {}
        for endpoint in ENDPOINTS:
            ep_results = optimise_endpoint(endpoint, feat_df, targets, split_indices, config, logger)
            if ep_results:
                all_results[endpoint] = ep_results

        save_json(all_results, "models/hyperparams.json")
        logger.info("Hyperparameter optimisation complete. Results saved to 'models/hyperparams.json'.")

        # Final test set evaluation (once, never again)
        logger.info("\n--- Final test set evaluation (performed exactly once) ---")
        final_eval = {}
        for endpoint, ep_results in all_results.items():
            y = targets[endpoint].dropna()
            X = feat_df.loc[y.index]
            test_idx = split_indices[endpoint]["test_indices"]
            test_idx = [i for i in test_idx if i in X.index]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]

            best_name = ep_results["selected_model"]
            best_model = joblib.load(f"models/best_model_{endpoint}.pkl")
            y_pred = best_model.predict(X_test)

            metrics = {
                "r2": float(r2_score(y_test.values, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test.values, y_pred))),
                "mae": float(np.mean(np.abs(y_test.values - y_pred))),
                "model_type": best_name,
            }
            final_eval[endpoint] = metrics
            logger.info(f"  {endpoint}: R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}")

        save_json(final_eval, "outputs/final_model_test_evaluation.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 7 — Hyperparameter Optimisation")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)
