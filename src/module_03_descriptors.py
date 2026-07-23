"""
Module 3 — RDKit Molecular Descriptor Generation

"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, save_json, set_seeds

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

DESCRIPTOR_NAMES = [
    "mw", "tpsa", "logp", "hbd", "hba", "rotatable_bonds",
    "aromatic_rings", "heavy_atom_count", "fraction_csp3",
]

#smiles correction for mod 2
'''df = pd.read_csv("data\\compounds\\pubchem_resolved.csv")
df = df.drop(columns=["smiles"])
df.to_csv("data\\compounds\\pubchem_resolved.csv")

new_column_data = ["CC1(C)CC2=C(C(=O)O)C(C)(C)CC3(C)C2C(=O)CC4(C)C3CC=C5C6(C)CCC(OC(=O)C(=C)CC)C(C)(C)C6CCC54C)C1", 
                   "CC(=O)OC1CCC2(C)C(CCC3(C)C2CC=C4C5(C)CCC(=O)C(C)(C)C5CCC43C)C1(C)C(=O)O",
                    "CC1(C)CC2=C(C(=O)O)C(C)(C)CC3(C)C2C(=O)CC4(C)C3CC=C5C6(C)CCC(OC(=O)C(=CC)C)C(C)(C)C6CCC54C)C1",
                    "CC=C(C)C(=O)OC1CCC2(C)C(CCC3(C)C2CC=C4C5(C)CCC(=O)C(C)(C)C5CCC43C)C1(C)C(=O)O",
                    "CC1(C)CC2=C(C(=O)O)C(C)(C)CC3(C)C2C(=O)CC4(C)C3CC=C5C6(C)CCC(O)C(C)(C)C6CCC54C)C1",
                    "C1=CC(=C(C=C1C2=CC(=O)C3=C(O2)C=C(C=C3O)O)O)O",
                    "COC1=C(C=C(C2=C1OC(=CC2=O)C3=CC=C(C=C3)O)O)O",
                    "C1=CC(=C(C=C1C2=C(C(=O)C3=C(O2)C=C(C=C3O)O)O)O)O",
                    "C1=CC(=CC=C1C2=C(C(=O)C3=C(O2)C=C(C=C3O)O)O)O",
                    "C1=CC(=C(C=C1C=CC(=O)O)O)O",
                    "C1C(C(C(CC1(C(=O)O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O",
                    "CC1=CCCC2(C)CC(C(=C)C2CC1)C(=C)C",
                    "CC1=CCC=C(C)CC=C(C)CC1",
                    "CC1=CC2CC1C2(C)C",
                    "CC1(C)C2CCC1(=C)CC2",
                    "CC(=CCCC(C)(C=C)O)C"
                ]
df.insert(4, 'smiles', new_column_data)
df.to_csv("data\\compounds\\pubchem_resolved.csv")'''

def smiles_to_descriptors(smiles: str) -> dict | None:
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit library unavailable. Ensure environment matches specification.")
    if not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
    }


def smiles_to_morgan_fp(smiles: str, radius: int = 2, nbits: int = 1024) -> np.ndarray | None:
    if not RDKIT_AVAILABLE or not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_tanimoto_matrix(fp_dict: dict) -> pd.DataFrame:
    compounds = list(fp_dict.keys())
    n = len(compounds)
    matrix = np.zeros((n, n))
    for i, a in enumerate(compounds):
        for j, b in enumerate(compounds):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                fp_a = fp_dict[a]
                fp_b = fp_dict[b]
                if fp_a is None or fp_b is None:
                    matrix[i, j] = matrix[j, i] = np.nan
                else:
                    intersection = int(np.bitwise_and(fp_a, fp_b).sum())
                    union = int(np.bitwise_or(fp_a, fp_b).sum())
                    tanimoto = intersection / union if union > 0 else 0.0
                    matrix[i, j] = matrix[j, i] = tanimoto
    return pd.DataFrame(matrix, index=compounds, columns=compounds)


def aggregate_descriptors(
    dataset_df: pd.DataFrame,
    descriptor_df: pd.DataFrame,
    compound_cols: list[str],
    method: str,
    logger,
) -> pd.DataFrame:
    desc_lookup = {}
    for _, row in descriptor_df.iterrows():
        col = row["compound_col"]
        desc_lookup[col] = {name: row.get(f"desc_{name}") for name in DESCRIPTOR_NAMES}

    aggregated_rows = []
    for idx, row in dataset_df.iterrows():
        available = {}
        for col in compound_cols:
            val = row.get(col)
            if pd.notna(val) and col in desc_lookup and desc_lookup[col].get("mw") is not None:
                available[col] = float(val)

        if not available:
            agg = {f"agg_{name}": np.nan for name in DESCRIPTOR_NAMES}
        else:
            abundances = np.array(list(available.values()))
            if method in ("weighted_mean", "abundance_weighted"):
                total = abundances.sum()
                weights = abundances / total if total > 0 else np.ones(len(abundances)) / len(abundances)
            elif method == "mean":
                weights = np.ones(len(abundances)) / len(abundances)
            else:
                weights = None

            agg = {}
            for name in DESCRIPTOR_NAMES:
                vals = np.array([desc_lookup[col].get(name, np.nan) for col in available])
                if method == "max":
                    agg[f"agg_{name}"] = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else np.nan
                else:
                    valid = ~np.isnan(vals)
                    if valid.any():
                        w = weights[valid]
                        w = w / w.sum()
                        agg[f"agg_{name}"] = float(np.dot(vals[valid], w))
                    else:
                        agg[f"agg_{name}"] = np.nan

        aggregated_rows.append(agg)

    return pd.DataFrame(aggregated_rows, index=dataset_df.index)


def run(config: dict, debug: bool = False) -> pd.DataFrame:
    ensure_dirs("data/descriptors")

    dataset_path = "data/processed/dataset.csv"
    resolved_path = "data/compounds/pubchem_resolved.csv"
    output_desc = "data/descriptors/rdkit_descriptors.csv"
    output_tanimoto = "data/descriptors/tanimoto_matrix.csv"
    output_agg = "data/descriptors/aggregated_descriptors.csv"

    with ModuleRunner(
        "module_03_descriptors",
        config,
        input_files=[dataset_path, resolved_path],
        output_files=[output_agg],
        debug=debug,
    ) as logger:
        set_seeds(config["RANDOM_SEED"])

        if not RDKIT_AVAILABLE:
            logger.warning("RDKit library path unresolvable; output arrays populated as stubs.")

        dataset_df = pd.read_csv(dataset_path)
        resolved_df = pd.read_csv(resolved_path)
        compound_cols = [c for c in dataset_df.columns if c.startswith("compound_")]

        descriptor_records = []
        fp_dict = {}
        for _, row in resolved_df.iterrows():
            col = row["compound_col"]
            smiles = row.get("smiles")
            rec = {"compound_col": col, "compound_name": row.get("compound_name"), "smiles": smiles}

            if RDKIT_AVAILABLE and pd.notna(smiles):
                desc = smiles_to_descriptors(smiles)
                if desc:
                    for name, val in desc.items():
                        rec[f"desc_{name}"] = val
                    fp = smiles_to_morgan_fp(smiles)
                    fp_dict[col] = fp
                else:
                    for name in DESCRIPTOR_NAMES:
                        rec[f"desc_{name}"] = np.nan
                    fp_dict[col] = None
            else:
                for name in DESCRIPTOR_NAMES:
                    rec[f"desc_{name}"] = np.nan
                fp_dict[col] = None

            descriptor_records.append(rec)

        descriptor_df = pd.DataFrame(descriptor_records)
        descriptor_df.to_csv(output_desc, index=False)

        valid_fps = {k: v for k, v in fp_dict.items() if v is not None}
        if len(valid_fps) >= 2:
            tanimoto_df = compute_tanimoto_matrix(valid_fps)
            tanimoto_df.to_csv(output_tanimoto)
        else:
            tanimoto_df = pd.DataFrame()

        annotations_path = Path("config/compound_annotations.yaml")
        compound_info = {}
        if annotations_path.exists():
            with open(annotations_path) as f:
                annotations = yaml.safe_load(f) or {}
            compound_info = annotations.get("compounds", {})

        class_groups: dict[str, list[str]] = {}
        for col in compound_cols:
            name_part = col.replace("compound_", "")
            cls = compound_info.get(name_part, {}).get("structural_class", "unknown")
            class_groups.setdefault(cls, []).append(col)

        for _, row in descriptor_df.iterrows():
            col = row["compound_col"]
            if pd.isna(row.get("desc_mw")):
                for cls, members in class_groups.items():
                    if col in members:
                        siblings = [m for m in members if m != col]
                        for name in DESCRIPTOR_NAMES:
                            sibling_vals = descriptor_df.loc[
                                descriptor_df["compound_col"].isin(siblings), f"desc_{name}"
                            ].dropna()
                            if not sibling_vals.empty:
                                descriptor_df.loc[descriptor_df["compound_col"] == col, f"desc_{name}"] = sibling_vals.mean()
                        break

        agg_method = config["FEATURES"]["descriptor_aggregation_method"]
        agg_df = aggregate_descriptors(dataset_df, descriptor_df, compound_cols, agg_method, logger)

        combined = pd.concat([dataset_df.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
        combined.to_csv(output_agg, index=False)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3 — RDKit Descriptor Generation")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)