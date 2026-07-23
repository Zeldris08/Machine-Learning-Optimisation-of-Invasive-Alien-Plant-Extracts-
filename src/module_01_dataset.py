"""
Module 1 — Dataset Construction  (multi-species, synthetic-augmented)

"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    InsufficientDataError,
    ModuleRunner,
    ensure_dirs,
    load_config,
    save_json,
    set_seeds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "study_id", "plant_species", "extraction_solvent",
    "concentration_mg_per_ml", "crop_species",
]

ENDPOINT_COLUMNS = ["germination_percent", "root_length_mm", "shoot_length_mm"]

TARGET_DONOR_SPECIES  = "lantana camara"
TARGET_RECEIVER_SPECIES = "zea mays"

# Typical w/v % → mg/mL conversion for plant extracts.
# A "10% w/v" extract = 100 mg/mL (10 g / 100 mL).
# Proportionally: X% w/v = X * 10 mg/mL.
PERCENT_TO_MGML = 10.0

# Species name normalisation — common abbreviations / aliases
SPECIES_ALIASES = {
    "l. camara":    "lantana camara",
    "l.camara":     "lantana camara",
    "lantanacamara":"lantana camara",
    "lantana camara l.": "lantana camara",
    "lantana camara linn.": "lantana camara",
    "lantana camara var. camara": "lantana camara",
    "p. hysterophorus": "parthenium hysterophorus",
    "oryza sativa":  "oryza sativa",
    "t. aestivum":  "triticum aestivum",
    "triticum aestivum l.": "triticum aestivum",
    "zea mays":     "zea mays",
    "h. scandens":  "humulus scandens",
    "a. indica":    "azadirachta indica",
    "a. retroflexus": "amaranthus retroflexus",
    "c. leptopetala": "choricarpia leptopetala",
    "c. album":     "chenopodium album",
}

# Canonical extraction solvent mapping
SOLVENT_MAP = {
    "distilled water": "aqueous",
    "water":           "aqueous",
    "aqueous":         "aqueous",
    "ethanol":         "ethanol",
    "80% ethanol":     "ethanol",
    "methanol":        "methanol",
    "85% methanol":    "methanol",
    "ethyl acetate":   "ethyl_acetate",
    "hexane":          "hexane",
    "n-hexane":        "hexane",
    "petroleum ether": "hexane",
    "chloroform":      "chloroform",
    "acetone":         "acetone",
}


def load_compound_class_map(annotations_path: str = "config/compound_annotations.yaml") -> dict:
    path = Path(annotations_path)
    if not path.exists():
        return {}
    with open(path) as f:
        annotations = yaml.safe_load(f) or {}
    compound_info = annotations.get("compounds", {})
    return {
        f"compound_{name}": meta.get("structural_class", "unknown")
        for name, meta in compound_info.items()
    }


COMPOUND_CLASS = load_compound_class_map()

# ---------------------------------------------------------------------------
# Step 1 — species & solvent normalisation
# ---------------------------------------------------------------------------

def normalise_species(name: str) -> str:
    if pd.isna(name) or not str(name).strip():
        return ""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return SPECIES_ALIASES.get(cleaned, cleaned)


def normalise_solvent(sol: str) -> str:
    if pd.isna(sol) or not str(sol).strip():
        return "aqueous"   # safest default for plant extract literature
    return SOLVENT_MAP.get(str(sol).strip().lower(), str(sol).strip().lower())


# ---------------------------------------------------------------------------
# Step 2 — rescue concentration from notes
# ---------------------------------------------------------------------------

def rescue_concentration_from_notes(row: pd.Series) -> float | None:
    """
    Parse the notes field for an explicit concentration value.
    Priority:
      1. Explicit mg/mL or g/L
      2. Explicit % w/v (converted via X * 10)
      3. Explicit % v/v (treated same as w/v for plant extracts)
    Returns None if nothing parseable is found.
    """
    notes = str(row.get("notes", "")) + " " + str(row.get("study_id", ""))
    notes_lower = notes.lower()

    # mg/mL or mg ml-1
    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*(?:mg[\s/]m[lL]|mg mL)", notes)
    if m:
        return float(m.group(1))

    # g/L  (÷ 1 since 1 g/L = 1 mg/mL)
    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*g[/\s]L", notes)
    if m:
        return float(m.group(1))

    # μg/mL → mg/mL
    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*[μu]g[\s/]m[lL]", notes)
    if m:
        return float(m.group(1)) / 1000.0

    # % w/v or % v/v — take the FIRST percentage mentioned
    # (usually "at X% concentration", not the result percentage)
    # Heuristic: look for "at \d+%" or "\d+% concentration"
    m = re.search(r"at\s+([\d]+(?:\.[\d]+)?)\s*%", notes_lower)
    if m:
        return float(m.group(1)) * PERCENT_TO_MGML

    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*%\s*(?:w/v|v/v|concentration|conc)", notes_lower)
    if m:
        return float(m.group(1)) * PERCENT_TO_MGML

    return None


# ---------------------------------------------------------------------------
# Step 3 — biologically grounded endpoint synthesis
# ---------------------------------------------------------------------------

# Seed for all RNG in this module — set once in run()
_rng: np.random.Generator = np.random.default_rng(42)

# Control (0 mg/mL) reference values per endpoint
CTRL_GERM    = 90.0   # % — typical well-watered petri dish control
CTRL_ROOT    = 80.0   # mm
CTRL_SHOOT   = 100.0  # mm

# Inflection point (IC50-like) and steepness for the Hill sigmoid
# Typical aqueous plant extract literature puts IC50 around 25–50 mg/mL
IC50_GERM    = 35.0
IC50_ROOT    = 30.0
IC50_SHOOT   = 30.0
HILL_N       = 1.8    # Hill coefficient — steepness

# Hormetic window: concentrations < this may stimulate rather than inhibit
HORMETIC_THRESHOLD = 5.0
HORMETIC_BOOST     = 0.08  # up to +8% stimulation


def _hill_inhibition(conc: float, ic50: float, n: float = HILL_N) -> float:
    """Fractional inhibition at given concentration (Hill equation). Returns 0–1."""
    if conc <= 0:
        return 0.0
    return (conc ** n) / (ic50 ** n + conc ** n)


def _hormetic_factor(conc: float) -> float:
    """Small stimulation at very low concentration (hormesis)."""
    if conc <= 0 or conc >= HORMETIC_THRESHOLD:
        return 0.0
    return HORMETIC_BOOST * np.sin(np.pi * conc / HORMETIC_THRESHOLD)


def synthesise_endpoints(
    conc: float,
    has_germ: bool,
    has_root: bool,
    has_shoot: bool,
    species_noise: float = 1.0,
) -> dict:
    """
    Return synthetic endpoint values for any combination of missing fields.
    `species_noise` is a per-species scaling factor (0.7–1.0) to introduce
    realistic biological variation across different receiver species.
    """
    inh_germ  = _hill_inhibition(conc, IC50_GERM)
    inh_root  = _hill_inhibition(conc, IC50_ROOT)
    inh_shoot = _hill_inhibition(conc, IC50_SHOOT)
    horm      = _hormetic_factor(conc)

    base_germ  = CTRL_GERM  * species_noise * (1.0 - inh_germ  + horm)
    base_root  = CTRL_ROOT  * species_noise * (1.0 - inh_root  + horm)
    base_shoot = CTRL_SHOOT * species_noise * (1.0 - inh_shoot + horm)

    # Biological noise: ~5% CV
    noise_germ  = _rng.normal(0, base_germ  * 0.05)
    noise_root  = _rng.normal(0, base_root  * 0.05)
    noise_shoot = _rng.normal(0, base_shoot * 0.05)

    result = {}
    if not has_germ:
        result["germination_percent"] = float(
            np.clip(base_germ + noise_germ, 0.0, 100.0))
    if not has_root:
        result["root_length_mm"] = float(
            np.clip(base_root + noise_root, 0.0, 200.0))
    if not has_shoot:
        result["shoot_length_mm"] = float(
            np.clip(base_shoot + noise_shoot, 0.0, 250.0))
    return result


# ---------------------------------------------------------------------------
# Step 4 — generate full synthetic training corpus
# ---------------------------------------------------------------------------

SYNTHETIC_SPECIES_PAIRS = [
    # (donor, plant_part, solvent, receiver, species_noise)
    ("lantana camara",           "leaf",       "aqueous",      "zea mays",             0.72),
    ("lantana camara",           "leaf",       "aqueous",      "zea mays",             0.68),
    ("lantana camara",           "leaf",       "aqueous",      "zea mays",             0.75),
    ("lantana camara",           "leaf",       "ethanol",      "zea mays",             0.70),
    ("lantana camara",           "leaf",       "aqueous",      "triticum aestivum",    0.78),
    ("lantana camara",           "leaf",       "aqueous",      "oryza sativa",         0.65),
    ("lantana camara",           "leaf",       "aqueous",      "oryza sativa",         0.80),
    ("lantana camara",           "leaf",       "methanol",     "lactuca sativa",       0.60),
    ("lantana camara",           "stem",       "aqueous",      "zea mays",             0.85),
    ("lantana camara",           "root",       "aqueous",      "triticum aestivum",    0.90),
    ("lantana camara",           "leaf",       "aqueous",      "phaseolus vulgaris",   0.82),
    ("lantana camara",           "leaf",       "ethyl_acetate","sorghum bicolor",      0.74),
    ("parthenium hysterophorus", "leaf",       "aqueous",      "zea mays",             0.70),
    ("parthenium hysterophorus", "whole plant","aqueous",      "triticum aestivum",    0.65),
    ("parthenium hysterophorus", "leaf",       "methanol",     "oryza sativa",         0.73),
    ("chromolaena odorata",      "leaf",       "aqueous",      "zea mays",             0.78),
    ("chromolaena odorata",      "leaf",       "aqueous",      "oryza sativa",         0.82),
    ("eucalyptus globulus",      "leaf",       "aqueous",      "hordeum vulgare",      0.60),
    ("eucalyptus globulus",      "leaf",       "aqueous",      "triticum aestivum",    0.65),
    ("ageratina adenophora",     "leaf",       "aqueous",      "oryza sativa",         0.55),
    ("acmella oleracea",         "aerial",     "ethanol",      "lactuca sativa",       0.68),
]

SYNTHETIC_CONCENTRATIONS = [0.0, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

SYNTHETIC_TPC_BY_SOLVENT = {
    "aqueous":      (15.0, 5.0),   # (mean, sd) mg GAE/g
    "ethanol":      (45.0, 10.0),
    "methanol":     (55.0, 12.0),
    "ethyl_acetate":(35.0, 8.0),
    "hexane":       (8.0,  3.0),
    "chloroform":   (12.0, 4.0),
    "acetone":      (30.0, 7.0),
}


def generate_synthetic_corpus(seed: int) -> pd.DataFrame:
    """
    Generate a biologically realistic synthetic training corpus.
    Each (species pair × concentration) combination gets 2–3 replicates,
    with per-replicate noise so CV folds don't see identical rows.
    """
    global _rng
    _rng = np.random.default_rng(seed)

    rows = []
    syn_idx = 0

    for donor, part, solvent, receiver, noise_factor in SYNTHETIC_SPECIES_PAIRS:
        for conc in SYNTHETIC_CONCENTRATIONS:
            n_reps = 2 if conc in (0.0, 10.0, 50.0) else 1
            for rep in range(n_reps):
                # Add tiny concentration jitter for non-zero rows to avoid
                # perfect duplicates across replicates
                conc_jitter = conc + (_rng.normal(0, conc * 0.02) if conc > 0 else 0)
                conc_jitter = max(0.0, conc_jitter)

                endpoints = synthesise_endpoints(
                    conc=conc_jitter,
                    has_germ=False,
                    has_root=False,
                    has_shoot=False,
                    species_noise=noise_factor,
                )

                tpc_mean, tpc_sd = SYNTHETIC_TPC_BY_SOLVENT.get(solvent, (20.0, 6.0))
                tpc = max(1.0, float(_rng.normal(tpc_mean, tpc_sd)))

                row = {
                    "study_id":                f"SYNTH_{syn_idx:04d}",
                    "doi":                     "",
                    "plant_species":           donor,
                    "plant_part":              part,
                    "extraction_solvent":      solvent,
                    "extraction_method":       "aqueous extraction" if solvent == "aqueous" else "solvent extraction",
                    "concentration_mg_per_ml": round(conc_jitter, 4),
                    "crop_species":            receiver,
                    "crop_variety":            "",
                    "germination_percent":     endpoints["germination_percent"],
                    "root_length_mm":          endpoints["root_length_mm"],
                    "shoot_length_mm":         endpoints["shoot_length_mm"],
                    "total_phenolic_content_mg_gae_per_g": round(tpc, 2),
                    "incubation_temp_c":       25.0,
                    "incubation_days":         7,
                    "tds_mg_per_ml":           round(conc_jitter * 0.05, 4),
                    "notes":                   f"Synthetic row — biologically grounded simulation (rep {rep+1})",
                }
                rows.append(row)
                syn_idx += 1

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Step 5 — clean and rescue real rows
# ---------------------------------------------------------------------------

def clean_real_rows(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Clean the real literature rows:
    - Normalise species names and solvents
    - Rescue concentration from notes
    - Fill missing required metadata with sensible defaults
    - Drop rows that are genuinely un-rescueable (no species, no endpoint,
      no concentration, AND no parseable info in notes)
    """
    # Drop totally blank rows (no species at all)
    df = df[df["plant_species"].notna() & (df["plant_species"].str.strip() != "")].copy()

    # Normalise species and solvent
    df["plant_species"] = df["plant_species"].apply(normalise_species)
    df["crop_species"]  = df["crop_species"].apply(normalise_species)
    df["extraction_solvent"] = df["extraction_solvent"].apply(normalise_solvent)

    # Rescue concentration from notes where missing
    conc_missing = df["concentration_mg_per_ml"].isna() | (df["concentration_mg_per_ml"] == 0)
    n_rescued = 0
    for idx in df.index[conc_missing]:
        rescued = rescue_concentration_from_notes(df.loc[idx])
        if rescued is not None and rescued > 0:
            df.at[idx, "concentration_mg_per_ml"] = rescued
            n_rescued += 1
    if n_rescued:
        logger.info(f"Rescued concentration from notes for {n_rescued} rows.")

    # Drop any row with concentration above 100 mg/mL
    high_conc = df["concentration_mg_per_ml"] > 100.0
    if high_conc.any():
        logger.info(f"Dropping {high_conc.sum()} rows with concentration > 100 mg/mL.")
        df = df[~high_conc].copy()

    # Fix negative endpoint values — convert to absolute (already logged upstream)
    for col in ["root_length_mm", "shoot_length_mm"]:
        if col in df.columns:
            neg = df[col] < 0
            if neg.any():
                logger.info(f"Converting {neg.sum()} negative values in '{col}' to absolute values.")
                df.loc[neg, col] = df.loc[neg, col].abs()

    # Drop the clearly junk row (row 138/140 — has "728.81" as plant species)
    junk = df["plant_species"].str.match(r"^[\d\.\,\s]+$", na=False)
    if junk.any():
        logger.info(f"Dropping {junk.sum()} rows with numeric garbage in plant_species.")
        df = df[~junk].copy()

    # Fill missing incubation defaults
    if "incubation_temp_c" in df.columns:
        df["incubation_temp_c"] = df["incubation_temp_c"].fillna(25.0)
    else:
        df["incubation_temp_c"] = 25.0

    if "incubation_days" in df.columns:
        df["incubation_days"] = df["incubation_days"].fillna(7).astype(int)
    else:
        df["incubation_days"] = 7

    # Fill missing total_phenolic_content with NaN (will be handled in Module 4)
    if "total_phenolic_content_mg_gae_per_g" not in df.columns:
        df["total_phenolic_content_mg_gae_per_g"] = np.nan

    # Fill missing crop_species with "unknown"
    df["crop_species"] = df["crop_species"].replace("", "unknown").fillna("unknown")

    # Drop rows that still have no concentration AND no endpoint after rescue
    no_conc = df["concentration_mg_per_ml"].isna() | (df["concentration_mg_per_ml"] == 0)
    no_ep   = df[ENDPOINT_COLUMNS].isna().all(axis=1)
    drop = no_conc & no_ep
    if drop.any():
        logger.info(f"Dropping {drop.sum()} rows with no concentration AND no endpoint after rescue.")
        df = df[~drop].copy()

    return df


def synthesise_missing_endpoints(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    For real rows that have a concentration but are missing some endpoints,
    synthesise the missing ones using the Hill dose-response model.
    """
    global _rng
    species_noise_map: dict[str, float] = {}

    def _get_noise(species: str) -> float:
        if species not in species_noise_map:
            # Each species gets a fixed noise factor for the run
            species_noise_map[species] = float(_rng.uniform(0.65, 0.95))
        return species_noise_map[species]

    n_synth_germ = n_synth_root = n_synth_shoot = 0

    for idx, row in df.iterrows():
        conc = row.get("concentration_mg_per_ml", 0.0)
        if pd.isna(conc):
            conc = 0.0
        noise = _get_noise(str(row.get("plant_species", "")))

        has_germ  = pd.notna(row.get("germination_percent"))
        has_root  = pd.notna(row.get("root_length_mm"))
        has_shoot = pd.notna(row.get("shoot_length_mm"))

        if has_germ and has_root and has_shoot:
            continue   # Nothing to synthesise

        synth = synthesise_endpoints(conc, has_germ, has_root, has_shoot, noise)
        for key, val in synth.items():
            df.at[idx, key] = val

        if not has_germ:
            n_synth_germ += 1
        if not has_root:
            n_synth_root += 1
        if not has_shoot:
            n_synth_shoot += 1

    logger.info(
        f"Synthesised missing endpoints for real rows: "
        f"germination={n_synth_germ}, root={n_synth_root}, shoot={n_synth_shoot}"
    )
    return df


# ---------------------------------------------------------------------------
# Compound column construction
# ---------------------------------------------------------------------------

# Canonical compound columns Module 2 needs to resolve via PubChem.
# These correspond exactly to the entries in compound_annotations.yaml.
CANONICAL_COMPOUNDS = [
    "lantadene_a", "lantadene_b", "lantadene_c", "lantadene_d",
    "icterogenin", "luteolin", "hispidulin", "quercetin", "kaempferol",
    "caffeic_acid", "chlorogenic_acid", "caryophyllene",
    "alpha_humulene", "alpha_pinene", "camphene", "linalool",
]

# Name aliases: maps whatever appears in compound_data JSON keys → canonical name
COMPOUND_NAME_ALIASES = {
    "(e)-caryophyllene":       "caryophyllene",
    "(e)-β-caryophyllene":     "caryophyllene",
    "e-beta-caryophyllene":    "caryophyllene",
    "caryophyllene":           "caryophyllene",
    "β-caryophyllene":         "caryophyllene",
    "alpha_humulene":          "alpha_humulene",
    "α-humulene":              "alpha_humulene",
    "humulene":                "alpha_humulene",
    "alpha_pinene":            "alpha_pinene",
    "α-pinene":                "alpha_pinene",
    "linalool":                "linalool",
    "camphene":                "camphene",
    "luteolin":                "luteolin",
    "luteolin-7-o-glucoside":  "luteolin",
    "quercetin":               "quercetin",
    "kaempferol":              "kaempferol",
    "caffeic_acid":            "caffeic_acid",
    "caffeic":                 "caffeic_acid",
    "chlorogenic_acid":        "chlorogenic_acid",
    "chlorogenic":             "chlorogenic_acid",
    "lantadene_a":             "lantadene_a",
    "lantadene_b":             "lantadene_b",
    "hispidulin":              "hispidulin",
}

# Per-species typical abundance profiles (0–100 scale, relative %).
# Used to fill compound columns for rows where compound_data is absent.
# Sources: published GC-MS / HPLC profiles in the allelopathy literature.
SPECIES_COMPOUND_PROFILES: dict[str, dict[str, float]] = {
    "lantana camara": {
        "caryophyllene":   18.0,  # dominant sesquiterpene in leaf oil
        "alpha_humulene":  12.0,
        "alpha_pinene":     5.0,
        "linalool":         3.0,
        "camphene":         2.5,
        "luteolin":         8.0,  # flavonoid from leaf extracts
        "hispidulin":       5.0,
        "quercetin":        4.0,
        "kaempferol":       3.0,
        "caffeic_acid":     6.0,
        "chlorogenic_acid": 7.0,
        "lantadene_a":     20.0,  # primary triterpenoid phytotoxin
        "lantadene_b":     12.0,
        "lantadene_c":      5.0,
        "lantadene_d":      3.0,
        "icterogenin":      4.0,
    },
    "parthenium hysterophorus": {
        "caffeic_acid":    10.0,
        "chlorogenic_acid": 8.0,
        "quercetin":        6.0,
        "luteolin":         4.0,
        "caryophyllene":    5.0,
    },
    "chromolaena odorata": {
        "quercetin":        8.0,
        "luteolin":         6.0,
        "caffeic_acid":     5.0,
        "alpha_pinene":     4.0,
        "caryophyllene":    7.0,
    },
    "eucalyptus globulus": {
        "alpha_pinene":    12.0,
        "camphene":         6.0,
        "linalool":         4.0,
        "caryophyllene":    8.0,
        "caffeic_acid":     5.0,
        "chlorogenic_acid": 9.0,
    },
    "ageratina adenophora": {
        "caffeic_acid":     7.0,
        "quercetin":        5.0,
        "caryophyllene":    6.0,
        "alpha_humulene":   4.0,
    },
    "acmella oleracea": {
        "caffeic_acid":     8.0,
        "luteolin":         7.0,
        "quercetin":        5.0,
    },
    # Generic fallback — moderate phenolic & terpenoid content
    "_default": {
        "caffeic_acid":     6.0,
        "chlorogenic_acid": 5.0,
        "quercetin":        4.0,
        "caryophyllene":    5.0,
        "alpha_pinene":     3.0,
    },
}


def _parse_compound_value(val) -> float | None:
    """
    Try to extract a numeric abundance (%) from whatever the JSON value is.
    Handles: plain numbers, "14.90%", "yes"/"detected" (→ 1.0), ranges "0.4–1.9%".
    Returns None for values that are clearly not abundance data.
    """
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if 0 <= float(val) <= 100:
            return float(val)
        return None
    s = str(val).strip().lower()
    if s in ("yes", "detected", "present", "identified", "isolated"):
        return 1.0          # presence-only; treat as low abundance
    if s in ("no", "absent", "nd", "not detected"):
        return 0.0
    # Handle ranges like "0.4–1.9%" — take the midpoint
    range_match = re.search(r"([\d.]+)\s*[–\-]\s*([\d.]+)", s)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        return (lo + hi) / 2.0
    # Plain percentage string "14.90%"
    pct_match = re.search(r"([\d.]+)\s*%", s)
    if pct_match:
        return float(pct_match.group(1))
    return None


def build_compound_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Create one numeric `compound_<name>` column per canonical compound for
    every row, so Module 2 has a well-defined set of columns to resolve via
    PubChem and Module 4 can compute class-abundance totals.

    Strategy:
      1. Parse `compound_data` JSON and map keys to canonical names where
         possible, extracting numeric abundance values.
      2. For rows where a canonical compound is still missing, fill with
         the species-typical profile (adding ±20% biological noise).
      3. For rows where the donor species has no profile entry, use the
         _default profile.
    """
    import json, ast

    # Initialise all canonical compound columns with NaN
    for cpd in CANONICAL_COMPOUNDS:
        col = f"compound_{cpd}"
        if col not in df.columns:
            df[col] = np.nan

    # --- Step 1: parse compound_data JSON ---
    parsed_count = 0
    if "compound_data" in df.columns:
        for idx, raw_val in df["compound_data"].items():
            if pd.isna(raw_val):
                continue
            s = str(raw_val).strip()
            if s in ("", "{}", "[]"):
                continue
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    continue
            if not isinstance(parsed, dict):
                continue

            for key, val in parsed.items():
                canonical = COMPOUND_NAME_ALIASES.get(key.lower().strip())
                if canonical is None:
                    continue
                numeric = _parse_compound_value(val)
                if numeric is not None:
                    col = f"compound_{canonical}"
                    # Only overwrite NaN — don't replace a value already set
                    if pd.isna(df.at[idx, col]):
                        df.at[idx, col] = numeric
                        parsed_count += 1

    logger.info(f"Parsed {parsed_count} compound abundance values from compound_data JSON.")

    # --- Step 2: fill remaining NaNs from species profile ---
    filled_count = 0
    for idx, row in df.iterrows():
        species = str(row.get("plant_species", "")).lower().strip()
        profile = SPECIES_COMPOUND_PROFILES.get(species,
                  SPECIES_COMPOUND_PROFILES["_default"])

        for cpd, base_abundance in profile.items():
            col = f"compound_{cpd}"
            if col in df.columns and pd.isna(df.at[idx, col]):
                # Add ±20% multiplicative noise for biological realism
                noise_factor = float(_rng.uniform(0.80, 1.20))
                df.at[idx, col] = round(base_abundance * noise_factor, 3)
                filled_count += 1

    # Remaining NaNs (compounds not in any profile) → 0.0
    for cpd in CANONICAL_COMPOUNDS:
        col = f"compound_{cpd}"
        df[col] = df[col].fillna(0.0)

    logger.info(
        f"Filled {filled_count} compound abundance values from species profiles. "
        f"All compound columns now fully populated."
    )
    return df


def detect_compound_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("compound_")]


def normalise_compound_columns(df: pd.DataFrame, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    present_cols = detect_compound_columns(df)
    if not present_cols:
        return df, pd.DataFrame()

    normalisation_records = []
    for col in present_cols:
        # Skip non-numeric columns (e.g. compound_data which holds raw JSON strings)
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Compound column '{col}' is non-numeric (dtype={df[col].dtype}); dropping it.")
            df = df.drop(columns=[col])
            normalisation_records.append({"compound": col, "original_max": None, "normalisation_factor": None, "dropped": True})
            continue

        col_max = df[col].max()
        if pd.isna(col_max) or col_max == 0:
            logger.warning(f"Compound column '{col}' has no valid positive values; skipping normalisation.")
            normalisation_records.append({"compound": col, "original_max": col_max, "normalisation_factor": np.nan, "dropped": False})
        else:
            df[col] = df[col] / col_max
            normalisation_records.append({"compound": col, "original_max": col_max, "normalisation_factor": 1.0 / col_max, "dropped": False})

    return df, pd.DataFrame(normalisation_records)


def impute_missing_values(df: pd.DataFrame, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputation_flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    present_compounds = detect_compound_columns(df)

    unannotated = [c for c in present_compounds if c not in COMPOUND_CLASS]
    if unannotated:
        logger.warning(
            f"{len(unannotated)} compound column(s) have no entry in "
            f"config/compound_annotations.yaml and will be treated as "
            f"structural_class='unknown': {unannotated}."
        )

    class_cols: dict[str, list[str]] = {}
    for col in present_compounds:
        cls = COMPOUND_CLASS.get(col, "unknown")
        class_cols.setdefault(cls, []).append(col)

    total_compound_imputations = 0
    for col in present_compounds:
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue
        cls = COMPOUND_CLASS.get(col, "unknown")
        class_sibling_cols = [c for c in class_cols.get(cls, []) if c != col and c in df.columns]

        for idx in df.index[missing_mask]:
            row_class_vals = df.loc[idx, class_sibling_cols].dropna()
            fill_val = row_class_vals.mean() if len(row_class_vals) > 0 else df[col].mean()
            df.at[idx, col] = fill_val
            imputation_flags.at[idx, col] = True
            total_compound_imputations += 1

    logger.info(f"Total compound abundance imputations: {total_compound_imputations}")
    return df, imputation_flags


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame, logger) -> pd.DataFrame:
    categorical_cols = {
        "plant_part":          "part",
        "extraction_solvent":  "solvent",
        "extraction_method":   "method",
    }
    for col, prefix in categorical_cols.items():
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(
            df[col].str.lower().str.replace(" ", "_", regex=False).fillna("unknown"),
            prefix=prefix, dtype=float
        )
        df = pd.concat([df, dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# Species / target-pair metadata
# ---------------------------------------------------------------------------

def add_species_metadata(df: pd.DataFrame, logger) -> pd.DataFrame:
    df["donor_is_target_species"] = df["plant_species"].str.contains(
        TARGET_DONOR_SPECIES, case=False, na=False
    )
    df["receiver_is_target_species"] = df["crop_species"].str.contains(
        TARGET_RECEIVER_SPECIES, case=False, na=False
    )

    cereal_crops = ["sorghum bicolor", "triticum aestivum", "oryza sativa", "zea mays"]
    df["crop_group"] = df["crop_species"].apply(
        lambda x: "cereal" if any(c in str(x).lower() for c in cereal_crops) else "other"
    )

    n_target = int((df["donor_is_target_species"] & df["receiver_is_target_species"]).sum())
    logger.info(
        f"Target pair (Lantana camara → Zea mays): {n_target} rows total."
    )
    return df


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame, config: dict, logger) -> None:
    if "germination_percent" in df.columns:
        germ = df["germination_percent"].dropna()
        out = germ[(germ < 0) | (germ > 100)]
        if not out.empty:
            logger.warning(f"Clipping {len(out)} germination_percent values outside [0, 100].")
            df["germination_percent"] = df["germination_percent"].clip(0, 100)

    for col in ["root_length_mm", "shoot_length_mm"]:
        if col in df.columns:
            neg = df[col] < 0
            if neg.any():
                df.loc[neg, col] = df.loc[neg, col].abs()

    if df["concentration_mg_per_ml"].isna().any():
        df["concentration_mg_per_ml"] = df["concentration_mg_per_ml"].fillna(0.0)

    min_pts = config["DATA"]["min_data_points"]
    if len(df) < min_pts:
        raise InsufficientDataError(
            f"Dataset has only {len(df)} rows; minimum required is {min_pts}."
        )
    logger.info(f"Quality checks passed. Final dataset: {len(df)} rows.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> dict:
    summary = {
        "total_rows":         int(len(df)),
        "unique_studies":     int(df["study_id"].nunique()) if "study_id" in df.columns else None,
        "unique_donor_species": int(df["plant_species"].nunique()),
        "concentration_range": {
            "min": float(df["concentration_mg_per_ml"].min()),
            "max": float(df["concentration_mg_per_ml"].max()),
        },
    }
    for ep in ENDPOINT_COLUMNS:
        if ep in df.columns:
            summary[f"{ep}_coverage"] = float(df[ep].notna().mean())
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: dict, debug: bool = False) -> pd.DataFrame:
    ensure_dirs("data/processed", "data/raw_literature")
    input_path  = config["DATA"]["raw_input_path"]
    output_path = "data/processed/dataset.csv"

    with ModuleRunner(
        "module_01_dataset",
        config,
        input_files=[input_path],
        output_files=[output_path],
        debug=debug,
    ) as logger:
        seed = config["RANDOM_SEED"]
        set_seeds(seed)
        global _rng
        _rng = np.random.default_rng(seed)

        # --- Load raw data ---
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Raw dataset not found at '{input_path}'.")
        raw = pd.read_csv(input_path)
        logger.info(f"Loaded {len(raw)} rows, {len(raw.columns)} columns.")

        # --- Clean real rows ---
        real_df = clean_real_rows(raw, logger)
        logger.info(f"Real rows after cleaning: {len(real_df)}")

        # --- Synthesise missing endpoints for real rows ---
        real_df = synthesise_missing_endpoints(real_df, logger)

        # --- Generate synthetic training corpus ---
        synth_df = generate_synthetic_corpus(seed)
        logger.info(f"Generated {len(synth_df)} synthetic training rows "
                    f"across {len(SYNTHETIC_SPECIES_PAIRS)} species pairs × "
                    f"{len(SYNTHETIC_CONCENTRATIONS)} concentrations.")

        # --- Combine ---
        combined = pd.concat([real_df, synth_df], ignore_index=True, sort=False)
        #combined = real_df

        # --- Species metadata & target-pair flags ---
        combined = add_species_metadata(combined, logger)

        # --- Build compound columns from compound_data JSON + species profiles ---
        combined = build_compound_columns(combined, logger)

        # --- Compound processing ---
        # Only relevant for real rows; synth rows have no compound columns
        combined, norm_df = normalise_compound_columns(combined, logger)
        if not norm_df.empty:
            norm_df.to_csv("data/raw_literature/compound_normalisation_factors.csv", index=False)

        combined, imputation_flags = impute_missing_values(combined, logger)
        imputation_flags.to_csv("data/processed/imputation_flags.csv")

        # --- Categorical encoding ---
        combined = encode_categoricals(combined, logger)

        # --- Final quality checks ---
        run_quality_checks(combined, config, logger)

        # --- Filter output columns strictly ---
        keep_columns = [
            "plant_species", "plant_part", "extraction_solvent", "extraction_method",
            "concentration_mg_per_ml", "crop_species", "crop_variety", "germination_percent",
            "root_length_mm", "shoot_length_mm", "total_phenolic_content_mg_gae_per_g",
            "compound_lantadene_a", "compound_lantadene_b", "compound_lantadene_c", "compound_lantadene_d",
            "compound_icterogenin", "compound_luteolin", "compound_hispidulin", "compound_quercetin",
            "compound_kaempferol", "compound_caffeic_acid", "compound_chlorogenic_acid", "compound_caryophyllene",
            "compound_alpha_humulene", "compound_alpha_pinene", "compound_camphene", "compound_linalool"
        ]
        # Drop columns if they don't exist in the final frame to avoid KeyErrors, though they should be present
        existing_keep_columns = [col for col in keep_columns if col in combined.columns]
        combined = combined[existing_keep_columns]

        # --- Save ---
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to '{output_path}'.")

        summary = compute_summary(combined)
        save_json(summary, "data/processed/dataset_summary.json")
        logger.info(f"Summary: {summary}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1 — Dataset Construction")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)