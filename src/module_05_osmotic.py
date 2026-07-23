"""
Module 5 — Osmotic Confound Handling

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, set_seeds

R = 8.314e-3   # Gas constant in kJ/(mol·K)
VAN_T_HOFF_FACTOR = 1.0   


def estimate_osmotic_water_potential(tds_mg_per_ml: float, temp_c: float = 25.0) -> float:
    """
    Estimates the osmotic water potential (MPa) from Total Dissolved Solids (mg/mL).
    Uses the van 't Hoff equation: Ψ = -iCRT.
    Assumes an average molecular weight of 200 g/mol for organic fractions.
    """
    if np.isnan(tds_mg_per_ml) or tds_mg_per_ml <= 0:
        return 0.0
    avg_mw = 200.0  
    c_mol_per_l = (tds_mg_per_ml / avg_mw)  
    c_molar = c_mol_per_l / 1000.0          
    temp_k = temp_c + 273.15
    
    psi_mpa = -1.0 * VAN_T_HOFF_FACTOR * c_molar * R * temp_k * 1000.0
    return float(np.clip(psi_mpa, -10.0, 0.0))


def osmotic_potential_to_inhibition_score(psi_mpa: float) -> float:
    """
    Translates osmotic water potential into an estimated root/germination inhibition proxy.
    Assumes negligible osmotic stress down to -0.2 MPa, followed by a monotonic decline.
    """
    if np.isnan(psi_mpa) or psi_mpa >= -0.2:
        return 0.0
    abs_psi = abs(psi_mpa)
    score = 1.0 - np.exp(-1.5 * (abs_psi - 0.2))
    return float(np.clip(score, 0.0, 1.0))


def fallback_proxy_osmotic(concentration_mg_per_ml: float) -> float:
    """
    Approximates osmotic correction metrics using concentration parameters 
    when measured TDS observations are absent.
    """
    estimated_tds = concentration_mg_per_ml * 0.08  
    psi = estimate_osmotic_water_potential(estimated_tds, 25.0)
    return osmotic_potential_to_inhibition_score(psi)


def run(config: dict, debug: bool = False) -> pd.DataFrame:
    ensure_dirs("data/features")
    feature_path = "data/features/feature_matrix.csv"
    full_matrix_path = "data/features/full_matrix_with_targets.csv"
    output_path = "data/features/osmotic_corrected_matrix.csv"

    with ModuleRunner(
        "module_05_osmotic",
        config,
        input_files=[feature_path, full_matrix_path],
        output_files=[output_path],
        debug=debug,
    ) as logger:
        set_seeds(config["RANDOM_SEED"])

        feat_df = pd.read_csv(feature_path)
        full_df = pd.read_csv(full_matrix_path)

        tds_col = "tds_mg_per_ml"
        temp_col = "incubation_temp_c"
        conc_col = "concentration_mg_per_ml"

        osmotic_scores = []
        is_approximate = []

        for idx in range(len(full_df)):
            tds = full_df.loc[idx, tds_col] if tds_col in full_df.columns else np.nan
            temp = full_df.loc[idx, temp_col] if temp_col in full_df.columns else 25.0
            conc = full_df.loc[idx, conc_col] if conc_col in full_df.columns else 0.0

            if pd.notna(tds) and tds > 0:
                psi = estimate_osmotic_water_potential(float(tds), float(temp))
                score = osmotic_potential_to_inhibition_score(psi)
                approx = False
            elif pd.notna(conc) and conc > 0:
                score = fallback_proxy_osmotic(float(conc))
                approx = True
            else:
                score = 0.0
                approx = False

            osmotic_scores.append(score)
            is_approximate.append(float(approx))

        feat_df["osmotic_inhibition_score"] = osmotic_scores
        feat_df["osmotic_correction_approximate"] = is_approximate

        approx_count = int(sum(is_approximate))
        if approx_count > 0:
            logger.warning(
                f"Osmotic correction computed via fallback approximation for {approx_count}/{len(feat_df)} observations "
                f"due to missing direct TDS data in field '{tds_col}'."
            )
        else:
            logger.info("Osmotic correction matrices constructed from experimental TDS data across all entries.")

        feat_df.to_csv(output_path, index=False)
        logger.info(f"Osmotic-corrected feature matrix written to '{output_path}'.")
        logger.info(f"Osmotic score distribution range: {min(osmotic_scores):.4f} to {max(osmotic_scores):.4f}")

    return feat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5 — Osmotic Confound Handling")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)