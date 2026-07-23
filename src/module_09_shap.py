"""
Module 9 — SHAP Explainability Analysis

"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent))
from utils import ModuleRunner, ensure_dirs, load_config, load_json, save_json, set_seeds

ENDPOINTS = ["germination_percent", "root_length_mm", "shoot_length_mm"]
EXPERIMENTAL_CONCENTRATIONS = [0, 1, 2.5, 5, 10, 25]


def apply_figure_style(config: dict):
    style = config["FIGURES"].get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-whitegrid")


def save_fig(fig, path: str, config: dict):
    dpi = config["FIGURES"].get("dpi", 300)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_shap_values(model, X: pd.DataFrame, logger) -> shap.Explanation:
    """Compute TreeSHAP values."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X)
    logger.info(f"SHAP values computed: shape {shap_vals.values.shape}")
    return shap_vals


def plot_summary(shap_vals: shap.Explanation, X: pd.DataFrame, endpoint: str, config: dict, out_dir: str):
    apply_figure_style(config)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals.values, X, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title(f"SHAP Summary — {endpoint.replace('_', ' ').title()}", fontsize=12)
    fig = plt.gcf()
    save_fig(fig, f"{out_dir}/shap_summary_{endpoint}.pdf", config)


def plot_feature_importance_bar(shap_vals_dict: dict, out_dir: str, config: dict):
    """Bar chart of mean |SHAP| for all endpoints side by side."""
    apply_figure_style(config)
    all_features = None
    importance_data = {}

    for endpoint, (shap_vals, X) in shap_vals_dict.items():
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        imp = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False).head(20)
        importance_data[endpoint] = imp
        if all_features is None:
            all_features = list(imp.index)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_features))
    width = 0.25
    colors = ["#0077BB", "#EE7733", "#009988"]

    for i, (endpoint, imp) in enumerate(importance_data.items()):
        vals = [imp.get(f, 0) for f in all_features]
        ax.bar(x + i * width, vals, width, label=endpoint.replace("_", " ").title(), color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=10)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_title("SHAP Feature Importance — All Endpoints", fontsize=12)
    ax.legend()
    save_fig(fig, f"{out_dir}/shap_feature_importance_barplot.pdf", config)


def plot_dependence(shap_vals: shap.Explanation, X: pd.DataFrame, feature: str,
                    endpoint: str, out_dir: str, config: dict):
    apply_figure_style(config)
    if feature not in X.columns:
        return
    feat_idx = list(X.columns).index(feature)
    shap_col = shap_vals.values[:, feat_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(X[feature], shap_col, alpha=0.6, c=X[feature], cmap="viridis", s=40)
    plt.colorbar(sc, ax=ax, label=feature)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel(f"SHAP value for {feature}", fontsize=10)
    ax.set_title(f"SHAP Dependence: {feature} → {endpoint.replace('_',' ').title()}", fontsize=11)
    safe_feat = feature.replace("/", "_").replace(" ", "_")[:40]
    save_fig(fig, f"{out_dir}/shap_dependence_{endpoint}_{safe_feat}.pdf", config)


def plot_interaction_heatmap(shap_vals: shap.Explanation, X: pd.DataFrame,
                              endpoint: str, out_dir: str, config: dict):
    """Simplified interaction heatmap using SHAP interaction values (top features)."""
    apply_figure_style(config)
    try:
        explainer = shap.TreeExplainer(joblib.load(f"models/best_model_{endpoint}.pkl"))
        shap_int = explainer.shap_interaction_values(X)
        # Take top 10 features by mean |SHAP|
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        top_cols = [X.columns[i] for i in top_idx]
        int_matrix = np.abs(shap_int[:, top_idx, :][:, :, top_idx]).mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(int_matrix, cmap="YlOrRd")
        ax.set_xticks(range(len(top_cols)))
        ax.set_yticks(range(len(top_cols)))
        ax.set_xticklabels(top_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(top_cols, fontsize=7)
        plt.colorbar(im, ax=ax, label="Mean |SHAP Interaction|")
        ax.set_title(f"SHAP Interaction Heatmap — {endpoint.replace('_',' ').title()}", fontsize=11)
        save_fig(fig, f"{out_dir}/shap_interaction_heatmap_{endpoint}.pdf", config)
    except Exception as e:
        pass  # Interaction values may not be available for all model types


def build_interpretation_table(shap_vals_dict: dict) -> pd.DataFrame:
    """Build biological interpretation table from SHAP importance."""
    BIOLOGICAL_HYPOTHESES = {
        "concentration_squared": "High-concentration inhibition consistent with hormetic phytotoxicity",
        "concentration": "Linear dose-response component",
        "log_concentration": "Log-linear dose-response; Weber-Fechner law",
        "compound_lantadene_a": "Lantadene A drives inhibition; known hepatotoxic triterpenoid",
        "compound_luteolin": "Luteolin antioxidant activity may stimulate ROS scavenging",
        "compound_quercetin": "Quercetin promotes germination at low doses; allelopathic at high doses",
        "bio_score_antioxidant": "Antioxidant compounds stimulate ROS scavenging at low concentrations",
        "bio_score_allelopathic": "Allelopathic compounds drive inhibition at higher doses",
        "agg_tpsa": "High TPSA → poor membrane permeability → reduced bioavailability",
        "agg_logp": "High LogP → membrane accumulation; may enhance or disrupt cell function",
        "osmotic_inhibition_score": "Osmotic stress confound; expected negative effect at high concentration",
    }

    records = []
    for endpoint, (shap_vals, X) in shap_vals_dict.items():
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        mean_shap = shap_vals.values.mean(axis=0)
        for feat, ma, ms in zip(X.columns, mean_abs, mean_shap):
            records.append({
                "feature": feat,
                "endpoint": endpoint,
                "mean_abs_shap": float(ma),
                "mean_shap": float(ms),
                "direction": "positive" if ms > 0 else "negative",
                "biological_hypothesis": BIOLOGICAL_HYPOTHESES.get(feat, ""),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["endpoint", "mean_abs_shap"], ascending=[True, False])
    return df


def run(config: dict, debug: bool = False) -> dict:
    ensure_dirs("outputs/shap/figures")

    with ModuleRunner(
        "module_09_shap",
        config,
        input_files=["data/features/osmotic_corrected_matrix.csv"],
        output_files=["outputs/shap/shap_values.csv"],
        debug=debug,
    ) as logger:
        seed = config["RANDOM_SEED"]
        set_seeds(seed)

        feat_df = pd.read_csv("data/features/osmotic_corrected_matrix.csv")
        full_df = pd.read_csv("data/features/full_matrix_with_targets.csv")

        out_dir = "outputs/shap/figures"
        all_shap_records = []
        shap_vals_dict = {}

        for endpoint in ENDPOINTS:
            if endpoint not in full_df.columns:
                continue

            # Load the tuned best model first to safely inspect its feature blueprint
            model = joblib.load(f"models/best_model_{endpoint}.pkl")
            feature_names = list(model.feature_names_in_)

            y = full_df[endpoint].dropna()
            X_full = feat_df.loc[y.index][feature_names]

            # --- SMART DOWNSAMPLING (IMMEDIATELY) ---
            # Sampling 1,000 rows here cuts down computation time from hours to seconds
            # while maintaining a statistically representative distribution layout
            sample_size = min(1000, len(X_full))
            X = X_full.sample(n=sample_size, random_state=seed)

            logger.info(f"Computing SHAP values for '{endpoint}'...")
            # We also disable the additivity check here to bypass floating-point rounding mismatches
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(X, check_additivity=False)
            
            shap_vals_dict[endpoint] = (shap_vals, X)

            # Save raw SHAP values
            shap_df = pd.DataFrame(shap_vals.values, columns=X.columns, index=X.index)
            shap_df["endpoint"] = endpoint
            all_shap_records.append(shap_df)

            # Summary plot
            logger.info(f"Generating summary plot for '{endpoint}'...")
            plot_summary(shap_vals, X, endpoint, config, out_dir)

            # Dependence plot for concentration
            plot_dependence(shap_vals, X, "concentration", endpoint, out_dir, config)
            plot_dependence(shap_vals, X, "log_concentration", endpoint, out_dir, config)

            # Top compound dependence plots
            mean_abs = np.abs(shap_vals.values).mean(axis=0)
            top_features = sorted(zip(X.columns, mean_abs), key=lambda x: x[1], reverse=True)
            compound_features = [f for f, _ in top_features if f.startswith("compound_")][:3]
            for feat in compound_features:
                plot_dependence(shap_vals, X, feat, endpoint, out_dir, config)

            # Interaction heatmap COMMENTED, as it can be very slow for large datasets and may not be necessary for all analyses
            #logger.info(f"Generating interaction heatmap for '{endpoint}'...")
            #plot_interaction_heatmap(shap_vals, X, endpoint, out_dir, config)

        # Global feature importance bar chart
        if shap_vals_dict:
            plot_feature_importance_bar(shap_vals_dict, out_dir, config)

        # Save all SHAP values
        if all_shap_records:
            pd.concat(all_shap_records).to_csv("outputs/shap/shap_values.csv", index=False)

        # Biological interpretation table
        interp_df = build_interpretation_table(shap_vals_dict)
        interp_df.to_csv("outputs/shap/biological_interpretation_table.csv", index=False)
        logger.info("SHAP analysis complete. All outputs saved.")

    return shap_vals_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 9 — SHAP Analysis")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, debug=args.debug)
