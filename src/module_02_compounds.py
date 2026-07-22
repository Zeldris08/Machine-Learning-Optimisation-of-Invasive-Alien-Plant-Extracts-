"""
Module 2 — Phytochemical Compound Resolution

Resolves unique chemical nomenclature to structural records via the PubChem PUG REST API.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    ModuleRunner,
    ensure_dirs,
    load_config,
    save_json,
    set_seeds,
)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
PROPERTIES = "IsomericSMILES,InChI,InChIKey,MolecularFormula,MolecularWeight"

MANUAL_CAS = {
    "lantadene_a": "467-81-2",
    "lantadene_b": "77315-73-2",
    "icterogenin": "16846-24-5",
    "luteolin": "491-70-3",
    "hispidulin": "1447-88-7",
    "quercetin": "117-39-5",
    "kaempferol": "520-18-3",
    "caffeic_acid": "331-39-5",
    "chlorogenic_acid": "327-97-9",
    "caryophyllene": "87-44-5",
    "alpha_humulene": "6753-98-6",
    "alpha_pinene": "80-56-8",
    "camphene": "79-92-5",
    "linalool": "78-70-6",
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


def query_pubchem(name: str, delay: float, logger) -> dict | None:
    url = f"{PUBCHEM_BASE}/{requests.utils.quote(name)}/property/{PROPERTIES}/JSON"
    try:
        resp = requests.get(url, timeout=15)
        time.sleep(delay)
        if resp.status_code == 200:
            data = resp.json()
            props = data["PropertyTable"]["Properties"][0]
            return {
                "pubchem_cid": props.get("CID"),
                "smiles": props.get("IsomericSMILES"),
                "inchi": props.get("InChI"),
                "inchikey": props.get("InChIKey"),
                "molecular_formula": props.get("MolecularFormula"),
                "molecular_weight_pubchem": props.get("MolecularWeight"),
            }
        return None
    except Exception as e:
        logger.warning(f"PubChem connection error for '{name}': {e}")
        return None


def resolve_compound(compound_col: str, delay: float, logger) -> dict:
    base_name = compound_col.replace("compound_", "").replace("_", " ")
    structural_class = COMPOUND_CLASS.get(compound_col, "unknown")

    record = {
        "compound_col": compound_col,
        "compound_name": base_name,
        "structural_class": structural_class,
        "pubchem_cid": None,
        "smiles": None,
        "inchi": None,
        "inchikey": None,
        "molecular_formula": None,
        "molecular_weight_pubchem": None,
        "resolution_status": "UNRESOLVED",
        "resolution_method": None,
    }

    # Pipeline Level 1: Standard Search
    result = query_pubchem(base_name, delay, logger)
    if result:
        record.update(result)
        record["resolution_status"] = "RESOLVED"
        record["resolution_method"] = "direct_name"
        return record

    # Pipeline Level 2: Prefix Matching
    stripped = base_name.replace("alpha-", "").replace("beta-", "").replace("alpha ", "").replace("beta ", "").strip()
    if stripped != base_name:
        result = query_pubchem(stripped, delay, logger)
        if result:
            record.update(result)
            record["resolution_status"] = "RESOLVED"
            record["resolution_method"] = "stripped_name"
            return record

    # Pipeline Level 3: Cross-reference CAS Identifier
    cas = MANUAL_CAS.get(compound_col.replace("compound_", ""))
    if cas:
        result = query_pubchem(cas, delay, logger)
        if result:
            record.update(result)
            record["resolution_status"] = "RESOLVED"
            record["resolution_method"] = "cas_number"
            return record

    return record


def apply_unknown_strategy(resolved_df: pd.DataFrame, strategy: str, logger) -> pd.DataFrame:
    unresolved = resolved_df[resolved_df["resolution_status"] == "UNRESOLVED"]
    if not unresolved.empty:
        logger.info(f"Executing unresolved fallback strategy ('{strategy}') for {len(unresolved)} structures.")
    return resolved_df


def run(config: dict, debug: bool = False) -> pd.DataFrame:
    ensure_dirs("data/compounds")
    input_path = "data/processed/dataset.csv"
    output_path = "data/compounds/pubchem_resolved.csv"

    with ModuleRunner(
        "module_02_compounds",
        config,
        input_files=[input_path],
        output_files=[output_path],
        debug=debug,
    ) as logger:
        set_seeds(config["RANDOM_SEED"])

        df = pd.read_csv(input_path)
        compound_cols = [c for c in df.columns if c.startswith("compound_")]
        
        delay = config["COMPOUNDS"]["pubchem_request_delay_seconds"]
        strategy = config["COMPOUNDS"]["unknown_compound_strategy"]

        records = [resolve_compound(col, delay, logger) for col in compound_cols]
        resolved_df = pd.DataFrame(records)
        resolved_df = apply_unknown_strategy(resolved_df, strategy, logger)

        resolved_count = (resolved_df["resolution_status"] == "RESOLVED").sum()
        total = len(resolved_df)
        logger.info(f"Structural Resolution Metric: {resolved_count}/{total} entries resolved.")

        resolved_df.to_csv(output_path, index=False)
        
        summary = {
            "total_compounds": total,
            "resolved": int(resolved_count),
            "unresolved": int(total - resolved_count),
            "resolution_rate": float(resolved_count / total) if total > 0 else 0.0,
            "strategy_applied": strategy,
        }
        save_json(summary, "data/compounds/resolution_summary.json")

    return resolved_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2 — Compound Resolution")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)