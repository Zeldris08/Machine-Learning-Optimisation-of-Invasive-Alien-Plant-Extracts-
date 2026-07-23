"""
Module 6 — Model Training and Comparison

"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, load_json, save_json, set_seeds

ENDPOINTS = ["germination_percent", "root_length_mm", "shoot_length_mm"]


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Load feature matrix, targets, and donor species labels (if available)."""
    feat_df = pd.read_csv("data/features/osmotic_corrected_matrix.csv")
    full_df = pd.read_csv("data/features/full_matrix_with_targets.csv")

    target_cols = [c for c in ENDPOINTS if c in full_df.columns]
    targets = full_df[target_cols].copy()

    species = full_df["plant_species"].str.lower().str.strip() if "plant_species" in full_df.columns else None

    # Remove target columns from feature matrix if accidentally present
    feat_df = feat_df.drop(columns=[c for c in ENDPOINTS if c in feat_df.columns], errors="ignore")

    # FIX 2: Drop zero-variance/constant features to eliminate StandardScaler RuntimeWarnings
    feat_df = feat_df.loc[:, feat_df.nunique() > 1]

    return feat_df, targets, species


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float,
    seed: int,
    concentration_col: str = "concentration",
) -> tuple:
    """
    Stratified split on concentration bin to ensure concentration regimes
    are proportionally represented in train and test.
    """
    # Bin concentration for stratification
    if concentration_col in X.columns:
        bins = pd.cut(X[concentration_col], bins=3, labels=["low", "med", "high"])
    else:
        bins = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_fraction,
        random_state=seed,
        stratify=bins,
    )
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², RMSE, MAE."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_cross_validation(model, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, seed: int) -> dict:
    """Run k-fold CV and return mean/std of metrics."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
    cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, return_train_score=False)
    return {
        "cv_r2_mean": float(cv_results["test_r2"].mean()),
        "cv_r2_std": float(cv_results["test_r2"].std()),
        "cv_rmse_mean": float(-cv_results["test_neg_rmse"].mean()),
        "cv_rmse_std": float(cv_results["test_neg_rmse"].std()),
        "cv_mae_mean": float(-cv_results["test_neg_mae"].mean()),
        "cv_mae_std": float(cv_results["test_neg_mae"].std()),
    }


def run_leave_one_species_out_cv(
    model, X: pd.DataFrame, y: pd.Series, species: pd.Series, logger
) -> dict | None:
    """
    Leave-one-donor-species-out cross-validation generalisation stress test.
    """
    species_aligned = species.loc[y.index]
    counts = species_aligned.value_counts()
    eligible_species = counts[counts >= 2].index.tolist()

    if len(eligible_species) < 2:
        logger.info(
            "Leave-one-species-out CV skipped: fewer than 2 donor species have "
            "≥2 rows each."
        )
        return None

    mask = species_aligned.isin(eligible_species)
    X_eligible = X.loc[mask]
    y_eligible = y.loc[mask]
    groups = species_aligned.loc[mask]

    logo = LeaveOneGroupOut()
    per_species_results = []

    for train_idx, test_idx in logo.split(X_eligible, y_eligible, groups=groups):
        held_out_species = groups.iloc[test_idx].iloc[0]
        X_tr, X_te = X_eligible.iloc[train_idx], X_eligible.iloc[test_idx]
        y_tr, y_te = y_eligible.iloc[train_idx], y_eligible.iloc[test_idx]

        if len(y_tr) < 5:
            continue

        from sklearn.base import clone
        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred))) if len(y_te) > 0 else np.nan
        mae = float(mean_absolute_error(y_te, y_pred)) if len(y_te) > 0 else np.nan
        r2 = float(r2_score(y_te, y_pred)) if len(y_te) > 1 else np.nan

        per_species_results.append({
            "held_out_species": held_out_species,
            "n_held_out": int(len(y_te)),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

    if not per_species_results:
        return None

    rmses = [r["rmse"] for r in per_species_results if not np.isnan(r["rmse"])]
    logger.info(
        f"  LOSO CV: {len(per_species_results)} species held out in turn. "
        f"Mean RMSE across species = {np.mean(rmses):.3f} "
    )

    return {
        "per_species": per_species_results,
        "mean_rmse": float(np.mean(rmses)) if rmses else None,
        "std_rmse": float(np.std(rmses)) if rmses else None,
        "n_species_evaluated": len(per_species_results),
    }


def build_rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )


def build_xgb(seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
    )


def run(config: dict, debug: bool = False) -> dict:
    ensure_dirs("models", "outputs/figures")

    # FIX 1: Updated output_files list to match exactly what ModuleRunner expects and validates
    with ModuleRunner(
        "module_06_training",
        config,
        input_files=["data/features/osmotic_corrected_matrix.csv"],
        output_files=[
            "models/best_model_germination_percent.pkl",
            "models/best_model_root_length_mm.pkl",
            "models/best_model_shoot_length_mm.pkl",
        ],
        debug=debug,
    ) as logger:
        seed = config["RANDOM_SEED"]
        set_seeds(seed)
        test_frac = config["DATA"]["test_fraction"]
        cv_folds = config["MODELS"]["cv_folds"]

        feat_df, targets, species = load_data(config)
        logger.info(f"Feature matrix: {feat_df.shape}, Endpoints: {list(targets.columns)}")

        run_loso = config["MODELS"]["cv_strategy"] == "leave_one_species_out"
        if run_loso and species is None:
            logger.warning(
                "cv_strategy='leave_one_species_out' requested, but 'plant_species' "
                "column not found. Falling back to standard k-fold."
            )
            run_loso = False
        elif species is not None:
            n_species = species.nunique()
            logger.info(f"Donor species diversity: {n_species} unique species in dataset.")

        all_results = {}
        loso_results = {}
        split_indices = {}

        for endpoint in ENDPOINTS:
            if endpoint not in targets.columns:
                logger.info(f"Endpoint '{endpoint}' not in dataset; skipping.")
                continue

            y = targets[endpoint].dropna()
            X = feat_df.loc[y.index].copy()
            logger.info(f"\n--- Training for endpoint: {endpoint} ({len(y)} samples) ---")

            X_train, X_test, y_train, y_test = split_data(X, y, test_frac, seed)
            split_indices[endpoint] = {
                "train_indices": list(X_train.index),
                "test_indices": list(X_test.index),
            }

            # Fit and save scaler on training data only
            scaler = StandardScaler()
            scaler.fit(X_train)
            joblib.dump(scaler, f"models/feature_scaler_{endpoint}.pkl")
            
            # FIX 4: Kept the scaling transformation, but note that tree models bypass these 
            # and train directly on raw X_train / X_test for operational stability.
            X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            endpoint_results = {}

            for model_name, model in [("random_forest", build_rf(seed)), ("xgboost", build_xgb(seed))]:
                logger.info(f"Training {model_name}...")

                cv_metrics = run_cross_validation(model, X_train, y_train, cv_folds, seed)
                logger.info(
                    f"  CV RMSE: {cv_metrics['cv_rmse_mean']:.3f} ± {cv_metrics['cv_rmse_std']:.3f}"
                    f"  |  CV R²: {cv_metrics['cv_r2_mean']:.3f} ± {cv_metrics['cv_r2_std']:.3f}"
                )

                if species is not None and species.nunique() > 1:
                    loso = run_leave_one_species_out_cv(model, X, y, species, logger)
                    if loso:
                        loso_results.setdefault(endpoint, {})[model_name] = loso

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_metrics = compute_metrics(y_test.values, y_pred_test)
                logger.info(
                    f"  Test R²: {test_metrics['r2']:.3f}  "
                    f"RMSE: {test_metrics['rmse']:.3f}  "
                    f"MAE: {test_metrics['mae']:.3f}"
                )

                model_path = f"models/{model_name}_{endpoint}.pkl"
                joblib.dump(model, model_path)

                endpoint_results[model_name] = {**test_metrics, **cv_metrics, "model_path": model_path}

            # FIX 3: Robust Model Selection logic incorporating LOSO cross-validation values.
            # Selects the architecture that manages unseen species safely, falling back to standard CV if LOSO lacks data.
            rf_cv_rmse = endpoint_results["random_forest"]["cv_rmse_mean"]
            xgb_cv_rmse = endpoint_results["xgboost"]["cv_rmse_mean"]

            rf_score = loso_results.get(endpoint, {}).get("random_forest", {}).get("mean_rmse", rf_cv_rmse)
            xgb_score = loso_results.get(endpoint, {}).get("xgboost", {}).get("mean_rmse", xgb_cv_rmse)
            
            if xgb_score < rf_score and (rf_score - xgb_score) / rf_score > 0.05:
                best_name = "xgboost"
            else:
                best_name = "random_forest"
                
            endpoint_results["selected_model"] = best_name
            logger.info(f"Selected model for '{endpoint}': {best_name} (Based on Unseen Species Optimization)")

            all_results[endpoint] = endpoint_results

        for endpoint in all_results:
            best = all_results[endpoint]["selected_model"]
            src = f"models/{best}_{endpoint}.pkl"
            dst = f"models/best_model_{endpoint}.pkl"
            shutil.copy(src, dst)
            logger.info(f"Best model for '{endpoint}' → '{dst}'")

        save_json(all_results, "outputs/model_comparison.json")
        save_json(split_indices, "models/split_indices.json")
        logger.info("Model comparison results saved to 'outputs/model_comparison.json'.")

        if loso_results:
            save_json(loso_results, "outputs/leave_one_species_out_cv.json")
            logger.info("Leave-one-species-out CV results saved to 'outputs/leave_one_species_out_cv.json'.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 6 — Model Training")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)