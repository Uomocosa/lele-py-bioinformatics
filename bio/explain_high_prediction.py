"""
SHAP Analysis for High Capacity Predictions.

This script uses SHAP (SHapley Additive exPlanations) with KernelExplainer to
understand why a specific polymer/drug combination produces a high capacity prediction.

The script:
1. Loads a trained PeeSmileCapacityPredictor model
2. Compares two test cases:
   - High-prediction polymer: *Nc1ccc(NC(=O)c2ccc(C(=O)NNC(=O)c3ccc(*)cc3)cc2)cc1
   - Normal polymer: *CC(C)(C(=O)OCCO)*
3. Uses SHAP KernelExplainer to compute feature importance
4. Generates:
   - shap_summary.png: Beeswarm plot showing feature impact distribution
   - shap_summary_bar.png: Bar plot showing global feature importance
   - top_features.csv: Ranked list of features by SHAP value
   - feature_comparison.csv: Side-by-side comparison of high vs normal features
"""

import sys
import types
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np

import bio
from bio.Bioinformatics.PeeSmileCapacityPredictor import PeeSmileCapacityPredictor
from bio.Dataset import PDCCMethod
from bio.__global__ import PDCC_CSV, RESULTS_DIR, LOGURU_SIMPLE_FORMAT
from loguru import logger

SAVE_DIR = RESULTS_DIR / "explain_high_prediction"


def test_():
    main()


def main():
    logger.info("Loading trained model...")
    pscp = PeeSmileCapacityPredictor()
    model = pscp.load_trained_model()
    model.eval()

    high_pred_df = pd.DataFrame(
        {
            "POLYMER_USED": ["*Nc1ccc(NC(=O)c2ccc(C(=O)NNC(=O)c3ccc(*)cc3)cc2)cc1"],
            "DRUG": ["CN(C)C(=N)N=C(N)N"],
            "WATER_PH": [8.2],
            "CONCENTRATION": [12.5],
        }
    )

    normal_df = pd.DataFrame(
        {
            "POLYMER_USED": ["*CC(C)(C(=O)OCCO)*"],
            "DRUG": ["CN(C)C(=N)N=C(N)N"],
            "WATER_PH": [8.2],
            "CONCENTRATION": [12.5],
        }
    )

    high_pred = model.predict(high_pred_df)
    normal_pred = model.predict(normal_df)
    logger.info(f"High prediction: {high_pred:.2f}")
    logger.info(f"Normal prediction: {normal_pred:.2f}")

    logger.info("Preparing background data...")
    df = pd.read_csv(PDCC_CSV)
    df = PDCCMethod.increment_dataset(
        df,
        PDCCMethod.increment_dataset.Options(
            method="interpolate_then_add_origins", n_points=5
        ),
    )
    df = PDCCMethod.convert_names_to_smiles(df)
    background_sample = df.sample(min(100, len(df)), random_state=42).reset_index(
        drop=True
    )

    logger.info("Computing features...")
    background_features = model.featurize_fn(background_sample)
    high_features = model.featurize_fn(high_pred_df)
    normal_features = model.featurize_fn(normal_df)

    expected_n_features = (
        model.x_scaler.n_features_in_ if model.x_scaler is not None else None
    )

    feature_columns = [
        col for col in background_features.columns if col not in ["CAPACITY", "SOURCE"]
    ]
    if expected_n_features and len(feature_columns) > expected_n_features:
        logger.warning(
            f"Feature mismatch: got {len(feature_columns)} columns, expected {expected_n_features}. Truncating to expected."
        )
        feature_columns = feature_columns[:expected_n_features]
    elif expected_n_features and len(feature_columns) < expected_n_features:
        logger.warning(
            f"Feature mismatch: got {len(feature_columns)} columns, expected {expected_n_features}. Will use available columns."
        )

    common_cols = [
        col
        for col in feature_columns
        if col in high_features.columns and col in normal_features.columns
    ]
    logger.info(f"Using {len(common_cols)} features for SHAP analysis")

    background_numeric = (
        background_features[common_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(float)
    )
    high_numeric = (
        high_features[common_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(float)
    )

    logger.info("Computing SHAP values using KernelExplainer...")

    def model_predict(features):
        """Wrapper for SHAP that works directly with pre-computed features."""
        if hasattr(features, "values"):
            features = features.values
        features = np.array(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        x_tensor = torch.tensor(features, dtype=torch.float32)

        if model.x_scaler is not None:
            x_scaled = model.x_scaler.transform(x_tensor.numpy())
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            scaled_prediction = model.forward(x_tensor).cpu().numpy()

        if model.y_scaler is not None:
            prediction = model.y_scaler.inverse_transform(scaled_prediction)
        else:
            prediction = scaled_prediction

        return prediction.flatten()

    explainer = shap.KernelExplainer(model_predict, background_numeric)
    shap_values = explainer.shap_values(high_numeric)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, high_features, show=False)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "shap_summary.png", dpi=300)
    plt.close()
    logger.info(f"Saved shap_summary.png")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, high_features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "shap_summary_bar.png", dpi=300)
    plt.close()
    logger.info(f"Saved shap_summary_bar.png")

    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values.flatten()
    feature_importance = pd.DataFrame(
        {"feature": high_features.columns, "shap_value": shap_vals}
    ).sort_values("shap_value", ascending=False)

    feature_importance.to_csv(SAVE_DIR / "top_features.csv", index=False)
    logger.info(f"Saved top_features.csv")

    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 FEATURES DRIVING HIGH PREDICTION:")
    logger.info("=" * 60)
    print(feature_importance.head(20).to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: High vs Normal prediction features")
    logger.info("=" * 60)

    comp_df = pd.DataFrame(
        {
            "feature": high_features.columns,
            "high_value": high_features.values[0],
            "normal_value": normal_features.values[0],
            "shap_value": shap_vals,
        }
    )
    comp_df["diff"] = comp_df["high_value"] - comp_df["normal_value"]
    comp_df = comp_df.sort_values("shap_value", ascending=False).head(20)
    comp_df.to_csv(SAVE_DIR / "feature_comparison.csv", index=False)
    logger.info(f"Saved feature_comparison.csv")

    print(comp_df.to_string(index=False))

    logger.info(f"\nAll outputs saved to: {SAVE_DIR}")
