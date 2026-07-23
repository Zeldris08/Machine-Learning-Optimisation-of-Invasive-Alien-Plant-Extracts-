"""
Module 11 — Experimental Validation and Final Performance Analysis

"""

import argparse
import sys
import json
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DataLeakageError,
    ModuleRunner,
    PredictionTamperingError,
    ensure_dirs,
    hash_dict,
    hash_file,
    load_config,
    save_json,
    set_seeds,
)

def load_json(path: str) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


PREDICTED_ENDPOINTS  = ["germination_percent", "root_length_mm", "shoot_length_mm"]
VALIDATED_ENDPOINTS  = ["root_length_mm", "shoot_length_mm"]   # have wet-lab replicates
EXPERIMENTAL_CONCENTRATIONS = [0, 1, 2.5, 5, 10, 25]
TPC_CONCENTRATIONS   = [0, 0.5, 1, 2.5]   # Folin assay only run for these

GROWTH_REPLICATE_COLUMNS = {
    "root_length_mm":  ["root_length_mm_r1",  "root_length_mm_r2",  "root_length_mm_r3"],
    "shoot_length_mm": ["shoot_length_mm_r1",  "shoot_length_mm_r2", "shoot_length_mm_r3"],
}
TPC_REPLICATE_COLUMNS = ["tpc_mg_gae_per_g_r1", "tpc_mg_gae_per_g_r2", "tpc_mg_gae_per_g_r3"]


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------

def verify_locked_predictions(manifest: dict, logger) -> dict:
    stored_hash = manifest.get("prediction_hash", "").replace("sha256:", "")
    lock_path   = manifest.get("prediction_file", "")

    # If the path is empty or the file doesn't exist, look for the backup record
    if not lock_path or not Path(lock_path).exists():
        # Look for lock_record.json specifically
        locked_files = list(Path("outputs/locked").glob("lock_record.json"))
        
        if not locked_files:
            raise FileNotFoundError(
                "No locked prediction file found! Checked manifest entry and "
                "could not find 'outputs/locked/lock_record.json'."
            )
            
        lock_path = str(locked_files[0])
        logger.warning(f"Manifest prediction path was empty/invalid. Using found lock file: {lock_path}")

    prediction_object = load_json(lock_path)
    recomputed_hash   = hash_dict(prediction_object)

    if stored_hash and recomputed_hash != stored_hash:
        raise PredictionTamperingError(
            f"PREDICTION FILE HASH MISMATCH!\n"
            f"  Stored:     {stored_hash}\n"
            f"  Recomputed: {recomputed_hash}\n"
            f"  File: {lock_path}\n"
            f"The locked predictions file may have been modified."
        )

    logger.info(f"Hash verification PASSED. Hash: {recomputed_hash}")
    return prediction_object

# ---------------------------------------------------------------------------
# Load experimental data
# ---------------------------------------------------------------------------

def load_experimental(path: str, logger) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Experimental results not found at '{path}'.\n"
            "Required columns:\n"
            "  concentration_mg_per_ml\n"
            "  root_length_mm_r1, root_length_mm_r2, root_length_mm_r3\n"
            "  shoot_length_mm_r1, shoot_length_mm_r2, shoot_length_mm_r3\n"
            "Optional:\n"
            "  tpc_mg_gae_per_g_r1, tpc_mg_gae_per_g_r2, tpc_mg_gae_per_g_r3\n"
            "  (only fill for 0, 0.5, 1, 2.5 mg/mL — leave blank for 5 and 25)"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "concentration_mg_per_ml" not in df.columns:
        raise ValueError("Column 'concentration_mg_per_ml' missing from experimental CSV.")

    # Growth replicates — required
    for endpoint, rep_cols in GROWTH_REPLICATE_COLUMNS.items():
        missing = [c for c in rep_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Required replicate columns missing for '{endpoint}': {missing}"
            )

    # TPC — optional
    missing_tpc = [c for c in TPC_REPLICATE_COLUMNS if c not in df.columns]
    if missing_tpc:
        logger.warning(
            f"TPC columns not found: {missing_tpc}. TPC will be skipped."
        )
    else:
        tpc_present = df[TPC_REPLICATE_COLUMNS].notna().any(axis=1)
        logger.info(
            f"TPC data present for concentrations: "
            f"{sorted(df.loc[tpc_present, 'concentration_mg_per_ml'].tolist())}"
        )

    exp_concs    = set(df["concentration_mg_per_ml"].dropna().values)
    missing_c    = set(EXPERIMENTAL_CONCENTRATIONS) - exp_concs
    unexpected_c = exp_concs - set(EXPERIMENTAL_CONCENTRATIONS)
    if missing_c:
        logger.warning(f"Concentrations predicted but not measured: {sorted(missing_c)}")
    if unexpected_c:
        logger.warning(f"Extra concentrations in CSV (will be ignored): {sorted(unexpected_c)}")

    logger.info(f"Loaded {len(df)} rows, concentrations: {sorted(exp_concs)}")
    return df


# ---------------------------------------------------------------------------
# Replicate statistics
# ---------------------------------------------------------------------------

def compute_replicate_stats(exp_df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Compute mean and SD across replicates for each row.
    """
    has_tpc = all(c in exp_df.columns for c in TPC_REPLICATE_COLUMNS)
    rows = []

    for _, row in exp_df.iterrows():
        rec = {"concentration_mg_per_ml": row["concentration_mg_per_ml"]}

        # Growth endpoints
        for endpoint, rep_cols in GROWTH_REPLICATE_COLUMNS.items():
            vals = pd.to_numeric(row[rep_cols], errors="coerce").dropna()
            rec[f"{endpoint}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            rec[f"{endpoint}_sd"]   = float(vals.std())  if len(vals) > 1 else 0.0

        # TPC — NaN where not measured is expected and fine
        if has_tpc:
            tpc_vals = pd.to_numeric(row[TPC_REPLICATE_COLUMNS], errors="coerce").dropna()
            rec["tpc_mg_gae_per_g_mean"] = float(tpc_vals.mean()) if len(tpc_vals) > 0 else np.nan
            rec["tpc_mg_gae_per_g_sd"]   = float(tpc_vals.std())  if len(tpc_vals) > 1 else np.nan
        else:
            rec["tpc_mg_gae_per_g_mean"] = np.nan
            rec["tpc_mg_gae_per_g_sd"]   = np.nan

        rows.append(rec)

    stats_df = pd.DataFrame(rows)

    for col_base in list(GROWTH_REPLICATE_COLUMNS.keys()) + ["tpc_mg_gae_per_g"]:
        mean_col = f"{col_base}_mean"
        if mean_col not in stats_df.columns:
            continue
        valid = stats_df[mean_col].notna()
        if valid.any():
            lo = stats_df.loc[valid, mean_col].min()
            hi = stats_df.loc[valid, mean_col].max()
            logger.info(
                f"  {col_base}: {valid.sum()}/{len(stats_df)} concentrations "
                f"observed. Range {lo:.2f} – {hi:.2f}"
            )
        else:
            logger.info(f"  {col_base}: no observed values.")

    return stats_df


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_validation_metrics(
    pred:           list,
    obs:            list,
    pi_lower:       list,
    pi_upper:       list,
    concentrations: list,
    endpoint:       str,
) -> dict:
    pred     = np.array(pred,           dtype=float)
    obs      = np.array(obs,            dtype=float)
    pi_lower = np.array(pi_lower,       dtype=float)
    pi_upper = np.array(pi_upper,       dtype=float)
    concs    = np.array(concentrations, dtype=float)

    valid   = ~np.isnan(obs) & ~np.isnan(pred)
    pred_v  = pred[valid]
    obs_v   = obs[valid]
    concs_v = concs[valid]

    if len(pred_v) < 2:
        return {
            "endpoint": endpoint,
            "n_points": int(valid.sum()),
            "note":     "fewer than 2 matched points",
        }

    ss_res   = np.sum((obs_v - pred_v) ** 2)
    ss_tot   = np.sum((obs_v - obs_v.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse     = float(np.sqrt(np.mean((pred_v - obs_v) ** 2)))
    mae      = float(np.mean(np.abs(pred_v - obs_v)))
    bias     = float(np.mean(pred_v - obs_v))
    coverage = float(np.mean(
        (obs_v >= pi_lower[valid]) & (obs_v <= pi_upper[valid])
    ))

    def _partial(mask):
        if mask.sum() < 2:
            return {}
        p, o = pred_v[mask], obs_v[mask]
        ss_r = np.sum((o - p) ** 2)
        ss_t = np.sum((o - o.mean()) ** 2)
        return {
            "r2":   float(1 - ss_r / ss_t) if ss_t > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean((p - o) ** 2))),
            "mae":  float(np.mean(np.abs(p - o))),
        }

    return {
        "endpoint":           endpoint,
        "r2":                 r2,
        "rmse":               rmse,
        "mae":                mae,
        "bias":               bias,
        "coverage_90pct":     coverage,
        "n_points":           int(valid.sum()),
        "low_concentration":  _partial(concs_v <= 5),
        "high_concentration": _partial(concs_v > 5),
    }


# ---------------------------------------------------------------------------
# Applicability domain
# ---------------------------------------------------------------------------

def _build_concentration_row(feat_df, concentration, feature_names):
    row = feat_df[feature_names].mean().to_frame().T.copy()
    if "concentration"         in row.columns: row["concentration"]         = concentration
    if "log_concentration"     in row.columns: row["log_concentration"]     = np.log1p(concentration)
    if "concentration_squared" in row.columns: row["concentration_squared"] = concentration ** 2
    if "concentration_cubed"   in row.columns: row["concentration_cubed"]   = concentration ** 3
    for i, (lo, hi) in enumerate([(-np.inf, 2.5), (2.5, 10.0), (10.0, np.inf)]):
        col = f"conc_bin_{'low med high'.split()[i]}"
        if col in row.columns:
            row[col] = 1.0 if lo < concentration <= hi else 0.0
    if concentration == 0 and "conc_bin_low" in row.columns:
        row["conc_bin_low"] = 1.0
    return row


def applicability_domain_assessment(
    feat_df, feature_names, train_indices, full_df,
    experimental_concentrations, logger,
):
    from sklearn.neighbors import NearestNeighbors

    X_query = pd.concat(
        [_build_concentration_row(feat_df, c, feature_names)
         for c in experimental_concentrations],
        ignore_index=True,
    )[feature_names].fillna(0).values

    def _assess(X_train, label):
        if len(X_train) < 3:
            logger.warning(f"AD ({label}): <3 reference rows, results unreliable.")
        try:
            inv       = np.linalg.pinv(X_train.T @ X_train)
            leverages = np.array([float(x @ inv @ x) for x in X_query])
            lev_thr   = 3 * X_train.shape[1] / len(X_train)
            out_lev   = (leverages > lev_thr).tolist()
        except Exception:
            leverages = [np.nan] * len(X_query)
            lev_thr   = np.nan
            out_lev   = [False] * len(X_query)

        k      = min(5, max(1, len(X_train)))
        nn     = NearestNeighbors(n_neighbors=k).fit(X_train)
        d, _   = nn.kneighbors(X_query)
        md     = d.mean(axis=1)
        nn2    = NearestNeighbors(n_neighbors=min(k+1, len(X_train))).fit(X_train)
        td, _  = nn2.kneighbors(X_train)
        tm     = td[:, 1:].mean(axis=1) if td.shape[1] > 1 else td[:, 0]
        dt     = float(np.percentile(tm, 95)) if len(tm) > 0 else np.nan
        out_knn= (md > dt).tolist() if not np.isnan(dt) else [True]*len(X_query)

        return {
            "n_reference_rows":       int(len(X_train)),
            "leverage_threshold":     float(lev_thr) if not np.isnan(lev_thr) else None,
            "knn_distance_threshold": dt if not np.isnan(dt) else None,
            "per_concentration": [
                {
                    "concentration_mg_per_ml": float(c),
                    "leverage":                float(leverages[i]) if not np.isnan(leverages[i]) else None,
                    "outside_leverage_domain": bool(out_lev[i]),
                    "mean_knn_distance":       float(md[i]),
                    "outside_knn_domain":      bool(out_knn[i]),
                }
                for i, c in enumerate(experimental_concentrations)
            ],
            "n_outside_leverage_domain": int(sum(out_lev)),
            "n_outside_knn_domain":      int(sum(out_knn)),
        }

    general = _assess(feat_df[feature_names].fillna(0).values[train_indices],
                      "full literature corpus")
    tp_result, n_tp = None, 0

    if ("donor_is_target_species" in full_df.columns and
            "receiver_is_target_species" in full_df.columns):
        mask  = (full_df["donor_is_target_species"].fillna(False)
                 & full_df["receiver_is_target_species"].fillna(False))
        tp_idx = [i for i in train_indices if i in full_df.index and mask.loc[i]]
        n_tp   = len(tp_idx)
        if n_tp >= 3:
            tp_result = _assess(feat_df[feature_names].fillna(0).values[tp_idx],
                                "Lantana camara → Zea mays only")
        else:
            logger.warning(f"Only {n_tp} target-pair rows — skipping target-pair AD check.")
    else:
        logger.warning("donor/receiver target flags not found — re-run Module 4.")

    if tp_result:
        n = tp_result["n_outside_knn_domain"]
        msg = (f"⚠  {n}/{len(experimental_concentrations)} concentrations outside Lantana→Maize domain."
               if n > 0 else
               f"✓ All {len(experimental_concentrations)} concentrations within Lantana→Maize domain.")
        (logger.warning if n > 0 else logger.info)(msg)

    return {
        "general_literature_corpus":   general,
        "target_pair_lantana_maize":   tp_result,
        "n_target_pair_training_rows": n_tp,
        "leverage_threshold":          general["leverage_threshold"],
        "knn_distance_threshold":      general["knn_distance_threshold"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: dict, debug: bool = False) -> dict:
    ensure_dirs("outputs/validation", "outputs/validation/figures")

    with ModuleRunner(
        "module_11_validation",
        config,
        input_files=[
            "data/experimental/experimental_results.csv",
            "outputs/locked/data_manifest.json",
        ],
        output_files=["outputs/validation/final_comparison.csv"],
        debug=debug,
    ) as logger:
        set_seeds(config["RANDOM_SEED"])

        # 1. Verify locked predictions
        manifest          = load_json("outputs/locked/data_manifest.json")
        prediction_object = verify_locked_predictions(manifest, logger)

        # 2. Load CSV and compute replicate stats
        exp_df   = load_experimental(
            "data/experimental/experimental_results.csv", logger
        )
        stats_df = compute_replicate_stats(exp_df, logger)

        # 3. Comparison table
        ENDPOINT_TO_STATS_KEY = {
            "germination_percent": None,           # no replicates in this study
            "root_length_mm":      "root_length_mm",
            "shoot_length_mm":     "shoot_length_mm",
        }

        comparison_rows        = []
        validation_metrics_all = {}

        # Use predictions_registry directly from the manifest
        predictions_registry = manifest.get("predictions_registry", {})

        for endpoint in PREDICTED_ENDPOINTS:
            # Look up the endpoint's flat dictionary from manifest
            locked_preds = predictions_registry.get(endpoint, {})
            if not locked_preds:
                logger.warning(f"No locked predictions for '{endpoint}'.")
                continue

            stats_key = ENDPOINT_TO_STATS_KEY.get(endpoint)
            mean_col  = f"{stats_key}_mean" if stats_key else None
            sd_col    = f"{stats_key}_sd"   if stats_key else None

            ep_rows = []
            for conc in EXPERIMENTAL_CONCENTRATIONS:
                obs_row = stats_df[stats_df["concentration_mg_per_ml"] == conc]

                # Dynamically build keys matching data_manifest.json format (e.g., conc_2.5_point)
                conc_float = float(conc)
                point_key  = f"conc_{conc_float}_point"
                lower_key  = f"conc_{conc_float}_lower_90pct"
                upper_key  = f"conc_{conc_float}_upper_90pct"

                if mean_col and not obs_row.empty and mean_col in obs_row.columns:
                    obs_mean = float(obs_row[mean_col].iloc[0])
                    obs_sd   = (float(obs_row[sd_col].iloc[0])
                                if sd_col and sd_col in obs_row.columns else np.nan)
                else:
                    obs_mean = np.nan
                    obs_sd   = np.nan

                ep_rows.append({
                    "endpoint":                endpoint,
                    "concentration_mg_per_ml": conc,
                    "predicted_point":         locked_preds.get(point_key),
                    "predicted_lower_90":      locked_preds.get(lower_key),
                    "predicted_upper_90":      locked_preds.get(upper_key),
                    "observed_mean":           obs_mean,
                    "observed_sd":             obs_sd,
                    "has_wet_lab_data":        not np.isnan(obs_mean),
                })

            comparison_rows.extend(ep_rows)

            ep_df = pd.DataFrame(ep_rows).dropna(
                subset=["predicted_point", "observed_mean"]
            )
            if len(ep_df) >= 2:
                metrics = compute_validation_metrics(
                    pred           = ep_df["predicted_point"].tolist(),
                    obs            = ep_df["observed_mean"].tolist(),
                    pi_lower       = ep_df["predicted_lower_90"].tolist(),
                    pi_upper       = ep_df["predicted_upper_90"].tolist(),
                    concentrations = ep_df["concentration_mg_per_ml"].tolist(),
                    endpoint       = endpoint,
                )
                validation_metrics_all[endpoint] = metrics
                logger.info(
                    f"{endpoint}: R²={metrics.get('r2', float('nan')):.3f}  "
                    f"RMSE={metrics.get('rmse', float('nan')):.3f}  "
                    f"Bias={metrics.get('bias', float('nan')):.3f}  "
                    f"Coverage={metrics.get('coverage_90pct', float('nan')):.2f}"
                )
            else:
                if endpoint in VALIDATED_ENDPOINTS:
                    logger.warning(
                        f"{endpoint}: only {len(ep_df)} matched rows — need ≥2. "
                        f"Check concentrations in CSV match locked values exactly."
                    )
                else:
                    logger.info(
                        f"{endpoint}: no wet-lab replicates — predictions "
                        f"recorded in table, metrics skipped."
                    )

        # 4. Save
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv("outputs/validation/final_comparison.csv", index=False)
        save_json(validation_metrics_all, "outputs/validation/validation_metrics.json")
        logger.info("Comparison table → outputs/validation/final_comparison.csv")

        # 5. TPC — partial observations only (concentrations 0, 0.5, 1, 2.5)
        tpc_col = "tpc_mg_gae_per_g_mean"
        if tpc_col in stats_df.columns:
            tpc_rows = stats_df[stats_df[tpc_col].notna()]
            if not tpc_rows.empty:
                tpc_out = tpc_rows[
                    ["concentration_mg_per_ml",
                     "tpc_mg_gae_per_g_mean",
                     "tpc_mg_gae_per_g_sd"]
                ].to_dict(orient="records")
                save_json(tpc_out, "outputs/validation/tpc_observed.json")
                logger.info(
                    f"TPC recorded at "
                    f"{sorted(tpc_rows['concentration_mg_per_ml'].tolist())} mg/mL "
                    f"→ outputs/validation/tpc_observed.json"
                )
            else:
                logger.info("TPC columns present but all NaN — nothing saved.")
        else:
            logger.info("No TPC columns in CSV — skipping TPC output.")

        # 6. Applicability domain
        try:
            feat_df       = pd.read_csv("data/features/osmotic_corrected_matrix.csv")
            full_df_ad    = pd.read_csv("data/features/full_matrix_with_targets.csv")
            feature_names = load_json("data/features/feature_names.json")
            feature_names = [f for f in feature_names if f in feat_df.columns]
            split_indices = load_json("models/split_indices.json")
            
            # FIX: Protective checks to ensure split_indices is populated as expected
            valid_endpoints = [ep for ep in PREDICTED_ENDPOINTS if ep in split_indices]
            if not valid_endpoints:
                raise KeyError("None of the predicted endpoints were found in split_indices.json")
                
            first_ep      = valid_endpoints[0]
            train_idx     = split_indices[first_ep]["train_indices"]
            exp_concs     = config["LOCK"]["experimental_concentrations"]
            ad_result     = applicability_domain_assessment(
                feat_df, feature_names, train_idx,
                full_df_ad, exp_concs, logger,
            )
            save_json(ad_result, "outputs/validation/applicability_domain_assessment.json")
            logger.info("AD assessment → outputs/validation/applicability_domain_assessment.json")
        except Exception as e:
            logger.warning(f"AD assessment failed: {e}")

        logger.info("Validation complete. Outputs in outputs/validation/.")
        return {"comparison": comparison_df, "metrics": validation_metrics_all}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 11 — Experimental Validation")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    run(cfg, debug=args.debug)