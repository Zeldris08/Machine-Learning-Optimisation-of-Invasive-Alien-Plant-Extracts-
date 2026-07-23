# Machine Learning Optimisation of Invasive Alien Plant Extracts as Sustainable Biostimulants for Crop Growth

Computational pipeline for the Eskom Expo for Young Scientists 2026 (Plant Sciences) project investigating whether a species-conditioned machine learning model, trained on literature-derived allelopathy data, can predict the dose-dependent effects of *Lantana camara* extract on wheat (*Triticum aestivum*) germination and seedling growth — validated against an independent, pre-registered wet-lab bioassay.

Full write-up: see the accompanying research report for background, method rationale, and discussion. This README documents the code only.

## Overview

The pipeline is organised as eleven sequential modules (`src/module_01_*` → `src/module_11_*`), each independently runnable and reading/writing to a shared `data/`, `models/`, and `outputs/` directory tree defined in `config/config.yaml`.

| Stage | Module | Purpose |
|---|---|---|
| 1 | `module_01_dataset.py` | Builds the training corpus: harmonises literature-mined bioassay records and applies a transparent, clearly flagged (`is_synthetic=True`) synthetic dose-response augmentation step to address sparse real data. |
| 2 | `module_02_compounds.py` | Resolves *L. camara* secondary metabolites to structural records via the PubChem PUG REST API (CIDs, InChIKeys, SMILES). |
| 3 | `module_03_descriptors.py` | Generates RDKit 1D/2D molecular descriptors (MW, logP, TPSA, HBD/HBA, etc.) and 2048-bit Morgan fingerprints for each compound. |
| 4 | `module_04_features.py` | Assembles the full feature matrix: concentration transforms, compound abundance/interaction terms, Tanimoto similarity, biological function scores, and experimental condition encoders. |
| 5 | `module_05_osmotic.py` | Estimates osmotic water potential (van 't Hoff approximation) to decouple non-specific osmotic effects from biochemical phytotoxicity. |
| 6 | `module_06_training.py` | Trains and compares Random Forest and XGBoost regressors for germination %, root length, and shoot length under 5-fold cross-validation. |
| 7 | `module_07_hyperopt.py` | Hyperparameter search (Optuna) for the candidate model architectures. |
| 8 | `module_08_uncertainty.py` | Computes split-conformal (90%) prediction intervals from held-out residual quantiles. |
| 9 | `module_09_shap.py` | SHAP explainability analysis and figure generation (summary plots, dependence plots, interaction heatmaps). |
| 10 | `module_10_lock.py` | Cryptographically hashes and timestamps the final predictions (SHA-256) before any wet-lab data is collected — the pre-experimental firewall. |
| 11 | `module_11_validation.py` | Compares locked predictions against empirical bioassay results and computes final validation metrics (R², RMSE, MAE, bias, interval coverage), with tamper/leakage checks against the lock record. |

`src/utils.py` holds shared infrastructure: config loading, reproducible seeding, file/dict hashing, and the `ModuleRunner` context manager used by every module for logging and I/O-hash tracking.

## Repository structure

```
.
├── src/                   # Pipeline modules (01–11) + utils.py
├── notebooks/              # Exploratory notebooks (data exploration, feature
│                            # engineering, model comparison, SHAP, final validation)
├── models/                 # Trained model artefacts, scalers, hyperparameters
├── outputs/
│   ├── predictions/          # Point predictions and conformal intervals
│   ├── locked/                # data_manifest.json, lock_record.json (hashed pre-registration)
│   ├── validation/             # Final comparison, validation metrics, AD assessment
│   ├── shap/                    # SHAP values and biological interpretation table
│   └── hyperparameter_search/
├── figures/                # Exported SHAP plots (PDF)
├── data/                   # Raw/processed datasets (not committed — see below)
└── config/                 # config.yaml (not committed — see below)
```

> **Note:** `data/` and `config/config.yaml` are not tracked in this repository (they contain machine-specific paths and, in the case of raw literature PDFs, third-party copyrighted content). See **Setup** below to recreate them.

## Setup

```bash
git clone https://github.com/Zeldris08/Machine-Learning-Optimisation-of-Invasive-Alien-Plant-Extracts-.git
cd Machine-Learning-Optimisation-of-Invasive-Alien-Plant-Extracts-
pip install -r requirements.txt   # see Dependencies below if this file is not yet present
```

Create `config/config.yaml` with, at minimum, the directory paths each module expects (`data_dir`, `models_dir`, `outputs_dir`) and a `seed` value (`42` was used for all reported results).

### Dependencies

The pipeline uses: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `rdkit`, `pyyaml`, `joblib`, `matplotlib`, `requests`. Data-mining scripts (`src/data_1.py`) additionally use `pdfplumber`, `beautifulsoup4`, `tqdm`, and `openai` (pointed at a local Ollama instance for LLM-based literature extraction). Pin these in a `requirements.txt` if one isn't already present in the repo.

## Running the pipeline

Each module is a standalone CLI script:

```bash
python src/module_01_dataset.py --config config/config.yaml
python src/module_02_compounds.py --config config/config.yaml
python src/module_03_descriptors.py --config config/config.yaml
python src/module_04_features.py --config config/config.yaml
python src/module_05_osmotic.py --config config/config.yaml
python src/module_06_training.py --config config/config.yaml
python src/module_07_hyperopt.py --config config/config.yaml
python src/module_08_uncertainty.py --config config/config.yaml
python src/module_09_shap.py --config config/config.yaml
python src/module_10_lock.py --config config/config.yaml
python src/module_11_validation.py --config config/config.yaml
```

Add `--debug` to any module for verbose logging. Modules must be run in order — each reads artefacts written by the previous stage.

## The pre-experimental firewall

The central methodological safeguard of this project is that **Module 10 locks and SHA-256-hashes all model predictions before the wet-lab bioassay is run or unblinded** (`outputs/locked/lock_record.json`, `data_manifest.json`). Module 11 checks the empirical results against this lock record and will raise a `PredictionTamperingError` or `DataLeakageError` if the manifest has been altered after the fact — this is what lets the validation in the report be reported as genuinely pre-registered rather than fitted in hindsight.

## Reproducibility

All stochastic steps (train/test split, cross-validation folds, synthetic data generation, hyperparameter search) are seeded (`seed=42`) via `utils.set_seeds`. `models/split_indices.json` and `models/hyperparams.json` record the exact split and tuned hyperparameters used to produce the results in the report.

## Notebooks

`notebooks/01`–`05` mirror the pipeline stages for exploratory/interactive use (data exploration, feature engineering, model comparison, SHAP analysis, final validation) but are not part of the reproducible pipeline — treat `src/` as the source of truth.

## Citation / attribution

If reusing this pipeline, please cite the accompanying Expo research report: *Machine Learning Optimisation of Invasive Alien Plant Extracts as Sustainable Biostimulants for Crop Growth* (Eskom Expo for Young Scientists 2026).
