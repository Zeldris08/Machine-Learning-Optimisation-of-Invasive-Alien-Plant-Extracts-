"""
Module 4 — Feature Engineering

Constructs the complete feature matrix containing:
  - Concentration terms (linear, log, squared, binned)
  - Compound abundance matrices
  - Aggregated molecular descriptor states
  - Interaction matrices (concentration × compound, compound × compound)
  - Tanimoto similarity metrics
  - Biological function scores
  - Experimental condition encoders
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, save_json, set_seeds

AGG_DESC_COLS = [
    "agg_mw", "agg_tpsa", "agg_logp", "agg_hbd", "agg_hba",
    "agg_rotatable_bonds", "agg_aromatic_rings", "agg_fraction_csp3",
]

BIO_FUNCTIONS = [
    "antioxidant", "ros_scavenger", "allelopathic",
    "auxin_modulation", "membrane_disruption", "antimicrobial", "enzyme_inhibition",
]


def detect_compound_columns(df: pd.DataFrame) -> list[str]:
    """Any column prefixed 'compound_' — works across any donor species, not just Lantana."""
    return [c for c in df.columns if c.startswith("compound_")]


def structural_class_groups(df: pd.DataFrame, annotations: dict) -> dict[str, list[str]]:
    """
    Build {structural_class: [compound_col, ...]} from compound_annotations.yaml,
    restricted to columns actually present in df. Compounds with no annotation
    fall into 'unannotated' rather than being silently dropped from class totals,
    so multi-species literature doesn't lose signal just because a compound
    hasn't been manually classified yet.
    """
    compound_info = annotations.get("compounds", {})
    present = detect_compound_columns(df)
    groups: dict[str, list[str]] = {}
    for col in present:
        name = col.replace("compound_", "")
        cls = compound_info.get(name, {}).get("structural_class", "unannotated")
        groups.setdefault(cls, []).append(col)
    return groups


def load_compound_annotations(annotations_path: str = "config/compound_annotations.yaml") -> dict:
    with open(annotations_path) as f:
        return yaml.safe_load(f)


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only columns that exist in df."""
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Concentration features
# ---------------------------------------------------------------------------

def build_concentration_features(df: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    col = config["FEATURES"]["concentration_column"]
    df["concentration"] = df[col]

    if config["FEATURES"]["use_log_concentration"]:
        df["log_concentration"] = np.log1p(df["concentration"])

    if config["FEATURES"]["use_squared_concentration"]:
        df["concentration_squared"] = df["concentration"] ** 2

    # Cubic only if dataset is large enough
    if len(df) > 80:
        df["concentration_cubed"] = df["concentration"] ** 3
        logger.info("Added concentration_cubed (dataset size > 80).")

    # Binning
    bins = [-np.inf, 2.5, 10.0, np.inf]
    labels = ["low", "medium", "high"]
    df["concentration_bin"] = pd.cut(df["concentration"], bins=bins, labels=labels)
    bin_dummies = pd.get_dummies(df["concentration_bin"], prefix="conc_bin", dtype=float)
    df = pd.concat([df, bin_dummies], axis=1)
    df.drop(columns=["concentration_bin"], inplace=True)

    logger.info("Concentration features built.")
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def build_interaction_features(df: pd.DataFrame, annotations: dict, config: dict, logger) -> pd.DataFrame:
    if not config["FEATURES"]["use_interaction_features"]:
        return df

    conc = df["concentration"]
    groups = structural_class_groups(df, annotations)

    # Class totals — works for any structural class present in the annotations
    # file (terpenoid, flavonoid, phenolic, ...), not just the three Lantana
    # classes that used to be hardcoded here. 'unannotated' compounds get their
    # own pooled total too, so they still contribute interaction signal even
    # before someone manually classifies them.
    class_total_cols = []
    for cls, cols in groups.items():
        total_col = f"{cls}_total"
        df[total_col] = df[cols].sum(axis=1)
        df[f"concentration_x_{total_col}"] = conc * df[total_col]
        class_total_cols.append(total_col)
    if not groups:
        logger.warning("No compound columns found; skipping class-total interaction features.")

    # Concentration × individual compound, for the compounds with the highest
    # mean abundance in THIS dataset (generic — adapts to whichever species'
    # compounds dominate, instead of a fixed Lantana shortlist).
    compound_cols = detect_compound_columns(df)
    if compound_cols:
        mean_abundance = df[compound_cols].mean().sort_values(ascending=False)
        top_compounds = list(mean_abundance.head(4).index)
        for cpd in top_compounds:
            df[f"concentration_x_{cpd}"] = conc * df[cpd]
        logger.info(f"Concentration × compound interactions built for top compounds by abundance: {top_compounds}")
    else:
        top_compounds = []

    # Concentration × aggregated descriptors
    if "agg_logp" in df.columns:
        df["concentration_x_agg_logp"] = conc * df["agg_logp"]
    if "agg_tpsa" in df.columns:
        df["concentration_x_agg_tpsa"] = conc * df["agg_tpsa"]
    if "log_concentration" in df.columns and "agg_logp" in df.columns:
        df["log_concentration_x_agg_logp"] = df["log_concentration"] * df["agg_logp"]

    # Compound × compound synergy features among the top-abundance compounds
    for i in range(len(top_compounds)):
        for j in range(i + 1, len(top_compounds)):
            a, b = top_compounds[i], top_compounds[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]

    # Pairwise class-total interactions (e.g. terpenoid_total × flavonoid_total)
    for i in range(len(class_total_cols)):
        for j in range(i + 1, len(class_total_cols)):
            a, b = class_total_cols[i], class_total_cols[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]

    logger.info("Interaction features built.")
    return df


# ---------------------------------------------------------------------------
# Tanimoto similarity feature
# ---------------------------------------------------------------------------

def build_tanimoto_feature(
    df: pd.DataFrame, annotations: dict, tanimoto_path: str, logger
) -> pd.DataFrame:
    known_bioactive = annotations.get("known_germination_regulators", [])
    known_cols = [f"compound_{c}" for c in known_bioactive if f"compound_{c}" in df.columns]

    if not known_cols:
        logger.warning("No known germination regulator columns found; skipping Tanimoto feature.")
        df["max_tanimoto_to_known_bioactive"] = np.nan
        return df

    if Path(tanimoto_path).exists():
        tanimoto_df = pd.read_csv(tanimoto_path, index_col=0)
        scores = []
        for col in known_cols:
            available_cols = [c for c in df.columns if c.startswith("compound_") and c in tanimoto_df.columns]
            if col in tanimoto_df.index and available_cols:
                sims = tanimoto_df.loc[col, available_cols].dropna()
                scores.append(sims.max() if not sims.empty else 0.0)
            else:
                scores.append(0.0)
        df["max_tanimoto_to_known_bioactive"] = max(scores) if scores else np.nan
        logger.info("Tanimoto similarity feature built from pre-computed matrix.")
    else:
        logger.warning("Tanimoto matrix not found; using fallback heuristic (sum of known bioactive abundances).")
        df["max_tanimoto_to_known_bioactive"] = df[known_cols].max(axis=1)

    return df


# ---------------------------------------------------------------------------
# Biological function score features
# ---------------------------------------------------------------------------

def build_bio_scores(df: pd.DataFrame, annotations: dict, config: dict, logger) -> pd.DataFrame:
    if not config["FEATURES"]["use_biological_function_features"]:
        return df

    compound_info = annotations.get("compounds", {})

    for func in BIO_FUNCTIONS:
        # Find all compound columns annotated with this function
        func_cols = []
        for cpd_name, meta in compound_info.items():
            col = f"compound_{cpd_name}"
            if col in df.columns and func in meta.get("functions", []):
                func_cols.append(col)

        if not func_cols:
            df[f"bio_score_{func}"] = 0.0
        else:
            # Abundance-weighted sum
            df[f"bio_score_{func}"] = df[func_cols].fillna(0.0).sum(axis=1)

    logger.info(f"Biological function score features built for: {BIO_FUNCTIONS}")
    return df


# ---------------------------------------------------------------------------
# Donor / receiver species features
# ---------------------------------------------------------------------------

def build_species_features(df: pd.DataFrame, logger, min_count: int = 3) -> pd.DataFrame:
    """
    One-hot encode donor plant species and receiver crop species as model
    features. This is what lets the model learn species-conditioned patterns
    (e.g. 'this terpenoid profile is inhibitory for cereals in general')
    instead of having species identity discarded as metadata, which would
    silently collapse a multi-species dataset down to 'concentration and
    chemistry only, species ignored'.

    Donor species with fewer than `min_count` rows are pooled into
    'species_donor_other' so a handful of one-off papers don't each get
    their own near-useless one-hot column. The receiver/crop side is usually
    less diverse and is encoded directly (Module 1 already builds crop_*
    dummies, so this only adds donor-species dummies).
    """
    if "plant_species" not in df.columns:
        logger.warning("'plant_species' column not found; skipping donor species features.")
        return df

    species_clean = df["plant_species"].str.lower().str.strip()
    counts = species_clean.value_counts()
    rare_species = set(counts[counts < min_count].index)

    grouped = species_clean.apply(lambda s: "other" if s in rare_species else s)
    grouped = grouped.str.replace(r"[^a-z0-9]+", "_", regex=True)

    dummies = pd.get_dummies(grouped, prefix="species_donor", dtype=float)
    df = pd.concat([df, dummies], axis=1)

    n_rare = int(species_clean.isin(rare_species).sum())
    logger.info(
        f"Donor species features: {dummies.shape[1]} columns "
        f"({len(counts) - len(rare_species)} species kept individually, "
        f"{len(rare_species)} rare species ({n_rare} rows) pooled into 'species_donor_other')."
    )
    return df


# ---------------------------------------------------------------------------
# Experimental condition features
# ---------------------------------------------------------------------------

def build_condition_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    for col in ["incubation_temp_c", "incubation_days"]:
        if col not in df.columns:
            logger.warning(f"Experimental condition column '{col}' not found; filling with default.")
            df[col] = 25.0 if "temp" in col else 7

    # One-hot encoded columns are already in df from Module 1
    condition_cols = [c for c in df.columns if c.startswith(("part_", "solvent_", "method_", "crop_"))]
    logger.info(f"Experimental condition features: {['incubation_temp_c', 'incubation_days'] + condition_cols}")
    return df


# ---------------------------------------------------------------------------
# Feature matrix assembly and scaling
# ---------------------------------------------------------------------------

def assemble_feature_matrix(df: pd.DataFrame, logger) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Select all engineered feature columns (excluding target variables and metadata).
    Returns (feature_df, feature_names, feature_categories).
    """
    exclude_patterns = [
        "study_id", "doi", "plant_species", "crop_species", "crop_variety",
        "extraction_solvent", "extraction_method", "plant_part",
        "germination_percent", "root_length_mm", "shoot_length_mm",
        "total_phenolic_content", "notes", "crop_group",
        "concentration_unit", "pure_compound_only",
        "total_dissolved_solids_mg_per_ml",
        "concentration_mg_per_ml",  
        "donor_is_target_species", "receiver_is_target_species",
    ]

    feature_cols = []
    for col in df.columns:
        if any(col == pat or col.startswith(pat) for pat in exclude_patterns):
            continue
        if df[col].dtype in [object]:
            continue
        feature_cols.append(col)

    feature_df = df[feature_cols].copy()

    # Categorise features
    categories = {}
    for col in feature_cols:
        if col.startswith("concentration"):
            categories[col] = "concentration"
        elif col.startswith("compound_"):
            categories[col] = "compound_abundance"
        elif col.startswith("agg_"):
            categories[col] = "molecular_descriptor"
        elif col.startswith("bio_score_"):
            categories[col] = "biological_function"
        elif col.startswith("conc_bin_"):
            categories[col] = "concentration_bin"
        elif col.startswith("species_donor_"):
            categories[col] = "donor_species"
        elif "_x_" in col:
            categories[col] = "interaction"
        elif col in ["incubation_temp_c", "incubation_days"]:
            categories[col] = "experimental_condition"
        elif col.startswith(("part_", "solvent_", "method_", "crop_")):
            categories[col] = "experimental_condition_encoded"
        else:
            categories[col] = "other"

    logger.info(f"Feature matrix assembled: {len(feature_cols)} features, {len(feature_df)} rows.")
    return feature_df, feature_cols, categories


def run(config: dict, debug: bool = False) -> pd.DataFrame:
    ensure_dirs("data/features", "models")

    input_path = "data/descriptors/aggregated_descriptors.csv"
    output_path = "data/features/feature_matrix.csv"

    with ModuleRunner(
        "module_04_features",
        config,
        input_files=[input_path],
        output_files=[output_path],
        debug=debug,
    ) as logger:
        set_seeds(config["RANDOM_SEED"])

        df = pd.read_csv(input_path)
        annotations = load_compound_annotations()

        # Build feature groups
        df = build_concentration_features(df, config, logger)
        df = build_interaction_features(df, annotations, config, logger)
        df = build_tanimoto_feature(df, annotations, "data/descriptors/tanimoto_matrix.csv", logger)
        df = build_bio_scores(df, annotations, config, logger)
        df = build_species_features(df, logger)
        df = build_condition_features(df, logger)

        # Assemble
        feature_df, feature_names, categories = assemble_feature_matrix(df, logger)

        # Fill remaining NaNs with column mean (conservative fallback)
        nan_counts = feature_df.isna().sum()
        for col in feature_df.columns:
            if nan_counts[col] > 0:
                feature_df[col] = feature_df[col].fillna(feature_df[col].mean())
                logger.debug(f"Filled {nan_counts[col]} NaNs in '{col}' with column mean.")

        # Save feature matrix (unscaled — scaler fitted in Module 6 after split)
        feature_df.to_csv(output_path, index=False)
        logger.info(f"Feature matrix saved to '{output_path}'.")

        # Save feature names and categories
        save_json(feature_names, "data/features/feature_names.json")
        save_json(
            {"feature_names": feature_names, "feature_categories": categories},
            "data/features/feature_engineering_log.json",
        )

        target_cols = [c for c in ["germination_percent", "root_length_mm", "shoot_length_mm"] if c in df.columns]
        meta_cols = [c for c in [
            "study_id", "concentration_mg_per_ml", "plant_species", "crop_species",
            "donor_is_target_species", "receiver_is_target_species",
        ] if c in df.columns]
        full_output = pd.concat([df[meta_cols + target_cols].reset_index(drop=True),
                                  feature_df.reset_index(drop=True)], axis=1)
        full_output.to_csv("data/features/full_matrix_with_targets.csv", index=False)

    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4 — Feature Engineering")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)
