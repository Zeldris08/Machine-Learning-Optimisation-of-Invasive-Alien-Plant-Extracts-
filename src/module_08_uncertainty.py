"""
Module 8 — Uncertainty Quantification

"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, load_json, save_json, set_seeds

ENDPOINTS = ["germination_percent", "root_length_mm", "shoot_length_mm"]
EXPERIMENTAL_CONCENTRATIONS = [0, 1, 2.5, 5, 10, 25]


def build_prediction_row(
    feat_df: pd.DataFrame,
    concentration: float,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build a synthetic feature row for a given concentration by taking the
    mean of all training rows and overriding concentration-derived features.
    """
    row = feat_df[feature_names].mean().to_frame().T.copy()

    # Override concentration features
    if "concentration" in row.columns:
        row["concentration"] = concentration
    if "log_concentration" in row.columns:
        row["log_concentration"] = np.log1p(concentration)
    if "concentration_squared" in row.columns:
        row["concentration_squared"] = concentration ** 2
    if "concentration_cubed" in row.columns:
        row["concentration_cubed"] = concentration ** 3

    # One-hot concentration bin
    bins = [(-np.inf, 2.5), (2.5, 10.0), (10.0, np.inf)]
    labels = ["low", "med", "high"]
    for i, (lo, hi) in enumerate(bins):
        col = f"conc_bin_{labels[i]}"
        if col in row.columns:
            row[col] = 1.0 if lo < concentration <= hi else 0.0
    # Special case: 0 mg/mL is "low"
    if concentration == 0 and "conc_bin_low" in row.columns:
        row["conc_bin_low"] = 1.0

    return row


def conformal_prediction(
    model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_pred: pd.DataFrame,
    coverage: float,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split conformal prediction using calibration residuals.
    Returns (point_predictions, lower_bounds, upper_bounds).
    """
    y_calib_pred = model.predict(X_calib)
    residuals = np.abs(y_calib.values - y_calib_pred)
    alpha = 1.0 - coverage
    q_alpha = np.quantile(residuals, coverage)

    y_pred = model.predict(X_pred)
    lower = y_pred - q_alpha
    upper = y_pred + q_alpha

    logger.info(f"Conformal quantile (q_{coverage:.0%}): {q_alpha:.3f}")
    return y_pred, lower, upper


def bootstrap_prediction(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    coverage: float,
    n_boot: int,
    seed: int,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap prediction intervals."""
    rng = np.random.RandomState(seed)
    boot_preds = []

    for b in range(n_boot):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_b = X_train.iloc[idx]
        y_b = y_train.iloc[idx]
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_b, y_b)
        boot_preds.append(m.predict(X_pred))

    boot_preds = np.array(boot_preds)
    alpha = (1 - coverage) / 2
    lower = np.quantile(boot_preds, alpha, axis=0)
    upper = np.quantile(boot_preds, 1 - alpha, axis=0)
    point = np.mean(boot_preds, axis=0)

    logger.info(f"Bootstrap ({n_boot} samples) prediction intervals computed.")
    return point, lower, upper


def run(config: dict, debug: bool = False) -> dict:
    ensure_dirs("outputs/predictions")

    with ModuleRunner(
        "module_08_uncertainty",
        config,
        input_files=["data/features/osmotic_corrected_matrix.csv"],
        output_files=[
            "outputs/predictions/point_predictions.csv",
            "outputs/predictions/prediction_intervals.csv",
        ],
        debug=debug,
    ) as logger:
        seed = config["RANDOM_SEED"]
        set_seeds(seed)

        method = config["UNCERTAINTY"]["method"]
        coverage = config["UNCERTAINTY"]["coverage_level"]
        exp_concs = config["LOCK"]["experimental_concentrations"]

        feat_df = pd.read_csv("data/features/osmotic_corrected_matrix.csv")
        full_df = pd.read_csv("data/features/full_matrix_with_targets.csv")
        split_indices = load_json("models/split_indices.json")

        point_records = []
        interval_records = []

        for endpoint in ENDPOINTS:
            if endpoint not in full_df.columns:
                continue

            # Load the tuned best model first to safely inspect its feature blueprint
            model = joblib.load(f"models/best_model_{endpoint}.pkl")
            feature_names = list(model.feature_names_in_)

            y = full_df[endpoint].dropna()
            X = feat_df.loc[y.index][feature_names]

            train_idx = [i for i in split_indices[endpoint]["train_indices"] if i in X.index]
            test_idx = [i for i in split_indices[endpoint]["test_indices"] if i in X.index]

            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_calib = X.loc[test_idx]
            y_calib = y.loc[test_idx]

            for conc in exp_concs:
                # Synthetic rows now conform explicitly to the true expected feature list
                X_pred = build_prediction_row(feat_df[feature_names], conc, feature_names)

                if method == "conformal":
                    point, lower, upper = conformal_prediction(
                        model, X_calib, y_calib, X_pred, coverage, logger
                    )
                elif method == "bootstrap":
                    point, lower, upper = bootstrap_prediction(
                        model, X_train, y_train, X_pred, coverage, n_boot=200, seed=seed, logger=logger
                    )
                else:
                    # quantile_regression_forest: fall back to conformal if QRF not set up
                    logger.warning(f"Method '{method}' not fully implemented; falling back to conformal.")
                    point, lower, upper = conformal_prediction(
                        model, X_calib, y_calib, X_pred, coverage, logger
                    )

                point_records.append({
                    "endpoint": endpoint,
                    "concentration_mg_per_ml": conc,
                    "predicted": float(point[0]),
                })
                interval_records.append({
                    "endpoint": endpoint,
                    "concentration_mg_per_ml": conc,
                    "predicted": float(point[0]),
                    f"lower_{int(coverage*100)}pct": float(lower[0]),
                    f"upper_{int(coverage*100)}pct": float(upper[0]),
                })

                logger.info(
                    f"  {endpoint} @ {conc} mg/mL: "
                    f"pred={point[0]:.2f}  "
                    f"[{lower[0]:.2f}, {upper[0]:.2f}] ({int(coverage*100)}% PI)"
                )

        point_df = pd.DataFrame(point_records)
        interval_df = pd.DataFrame(interval_records)

        point_df.to_csv("outputs/predictions/point_predictions.csv", index=False)
        interval_df.to_csv("outputs/predictions/prediction_intervals.csv", index=False)
        logger.info("Point predictions and prediction intervals saved.")

    return {"point": point_df, "intervals": interval_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 8 — Uncertainty Quantification")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)
