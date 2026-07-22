"""
Module 10 — Prediction Locking and Data Versioning

Generates cryptographic manifest registries and anchor signatures across model arrays 
and point/interval predictions to preserve system state integrity prior to verification.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    ModuleRunner,
    ensure_dirs,
    hash_dict,
    hash_file,
    load_config,
    load_json,
    save_json,
    set_seeds,
    utc_now_iso,
)

ENDPOINTS = ["germination_percent", "root_length_mm", "shoot_length_mm"]
MODEL_TYPE_LABELS = {
    "random_forest": "RandomForest_v1",
    "xgboost": "XGBoost_v1",
}


def run(config: dict, debug: bool = False) -> dict:
    ensure_dirs("outputs/locked", "logs")

    with ModuleRunner(
        "module_10_lock",
        config,
        input_files=[
            "outputs/predictions/point_predictions.csv",
            "outputs/predictions/prediction_intervals.csv",
        ],
        output_files=["outputs/locked/data_manifest.json"],
        debug=debug,
    ) as logger:
        set_seeds(config["RANDOM_SEED"])

        point_df = pd.read_csv("outputs/predictions/point_predictions.csv")
        interval_df = pd.read_csv("outputs/predictions/prediction_intervals.csv")

        pred_records = {}
        for endpoint in ENDPOINTS:
            ep_pt = point_df[point_df["endpoint"] == endpoint]
            ep_int = interval_df[interval_df["endpoint"] == endpoint]

            if ep_pt.empty:
                continue

            endpoint_map = {}
            for _, row in ep_pt.iterrows():
                conc = float(row["concentration_mg_per_ml"])
                endpoint_map[f"conc_{conc}_point"] = float(row["predicted"])

            for _, row in ep_int.iterrows():
                conc = float(row["concentration_mg_per_ml"])
                cols = [c for c in row.index if c.startswith("lower_") or c.startswith("upper_")]
                for c in cols:
                    endpoint_map[f"conc_{conc}_{c}"] = float(row[c])

            pred_records[endpoint] = endpoint_map

        comparison_path = Path("outputs/model_comparison.json")
        model_selection = {}
        if comparison_path.exists():
            comparison_data = load_json(str(comparison_path))
            for endpoint in ENDPOINTS:
                if endpoint in comparison_data:
                    sel = comparison_data[endpoint].get("selected_model", "unknown")
                    model_selection[endpoint] = MODEL_TYPE_LABELS.get(sel, sel)

        lock_ts = utc_now_iso()
        data_manifest = {
            "lock_timestamp_utc": lock_ts,
            "random_seed_anchor": config["RANDOM_SEED"],
            "model_architecture_selection": model_selection,
            "predictions_registry": pred_records,
        }

        manifest_path = "outputs/locked/data_manifest.json"
        save_json(data_manifest, manifest_path)

        pred_hash = hash_dict(data_manifest)
        logger.info(f"Cryptographic system signature created: {pred_hash}")

        artifact_hashes = {}
        target_artifacts = [
            ("matrix_features", "data/features/osmotic_corrected_matrix.csv"),
            ("matrix_full", "data/features/full_matrix_with_targets.csv"),
            ("predictions_point", "outputs/predictions/point_predictions.csv"),
            ("predictions_interval", "outputs/predictions/prediction_intervals.csv"),
        ]

        for endpoint in ENDPOINTS:
            target_artifacts.append((f"model_binary_{endpoint}", f"models/best_model_{endpoint}.pkl"))

        for label, path in target_artifacts:
            p = Path(path)
            if p.exists():
                artifact_hashes[label] = hash_file(str(p))
            else:
                artifact_hashes[label] = None

        lock_record = {
            "lock_hash": pred_hash,
            "lock_timestamp_utc": lock_ts,
            "artifact_manifest_hashes": artifact_hashes,
        }
        save_json(lock_record, "outputs/locked/lock_record.json")

        locking_event = (
            f"======================================================================\n"
            f"SYSTEM STATE CRYPTOGRAPHIC IMMUTABILITY LOCK RECORD\n"
            f"======================================================================\n"
            f"Timestamp (UTC):          {lock_ts}\n"
            f"Manifest Path:            {manifest_path}\n"
            f"System State Hash:        {pred_hash}\n"
            f"\n"
            f"The state hash registry logs should be explicitly filed to verify \n"
            f"reproducibility boundaries before system verification operations execute.\n"
            f"\n"
            f"Downstream Module Integrity Hashes:\n"
        )
        for label, h in artifact_hashes.items():
            locking_event += f"  {label}: {h}\n"

        lock_log_path = "logs/locking_event.log"
        with open(lock_log_path, "w") as f:
            f.write(locking_event)

        print("\n" + "=" * 70)
        print(" SYSTEM REGISTRY STATE LOCK COMPLETE")
        print("=" * 70)
        print(f"Timestamp: {lock_ts}")
        print(f"File:      {manifest_path}")
        print(f"SHA-256:   {pred_hash}")
        print("\nState hashes registered for pipeline auditing.")
        print("=" * 70 + "\n")

        return {"lock_path": manifest_path, "hash": pred_hash, "manifest": data_manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 10 — Prediction Locking")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)