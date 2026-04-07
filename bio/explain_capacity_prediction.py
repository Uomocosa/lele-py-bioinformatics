"""
SHAP Analysis for Capacity Predictions.

This script uses SHAP (SHapley Additive exPlanations) with KernelExplainer to
understand why specific polymer/drug combinations produce their predicted capacity values.

The script:
1. Loads a trained PeeSmileCapacityPredictor model
2. Accepts a list of polymers and a molecule name (e.g., "aspirin", "metformin")
3. Uses SHAP KernelExplainer to compute feature importance for each polymer
4. Generates comparison outputs across all polymers:
   - shap_summary.png: Combined beeswarm plot showing feature impact distribution
   - shap_summary_bar.png: Bar chart showing global feature importance
   - top_features.csv: Ranked list of features by SHAP value for each polymer
   - polymer_comparison.csv: Table comparing polymers by capacity and top features
"""

import tyro
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap
import torch
import yaml
from loguru import logger

import bio
from bio.Bioinformatics.PeeSmileCapacityPredictor import PeeSmileCapacityPredictor
from bio.Dataset import PDCCMethod
from bio.__global__ import PDCC_CSV, RESULTS_DIR

SAVE_DIR = RESULTS_DIR / "explain_capacity_prediction"


@dataclass
class Config:
    """Configuration for capacity prediction explanation using SHAP."""

    molecule_name: tyro.conf.Positional[str] = field(
        metadata={
            "help": "Drug molecule name (e.g., 'aspirin', 'metformin', 'lisinopril')"
        }
    )
    polymer_smiles: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of p-smiles polymers to analyze (can be specified multiple times)"
        },
    )
    water_ph: float = 8.2
    concentration: float = 12.5
    save_dir: Optional[Path] = SAVE_DIR


def main():
    bio.setup_loguru()
    config = tyro.cli(Config)
    explain_capacity_prediction(config)


def test_():
    bio.setup_loguru()
    config = Config(
        molecule_name="aspirin",
        polymer_smiles=["*Nc1ccc(NC(=O)c2ccc(C(=O)NNC(=O)c3ccc(*)cc3)cc2)cc1"],
        save_dir=bio.Bioinformatics.__global__.HELPER_DIR
        / "explain_capacity_prediction",
    )
    explain_capacity_prediction(config)


@pytest.mark.above10s
def test_aspirin_above_threshold():
    bio.setup_loguru()
    polymers = get_polymers_above_threshold("aspirin", threshold=50)
    config = Config(
        molecule_name="aspirin",
        polymer_smiles=polymers,
    )
    explain_capacity_prediction(config)


@pytest.mark.above10s
def test_lisinopril_above_threshold():
    bio.setup_loguru()
    polymers = get_polymers_above_threshold("lisinopril", threshold=50)
    config = Config(
        molecule_name="lisinopril",
        polymer_smiles=polymers,
    )
    explain_capacity_prediction(config)


@pytest.mark.above10s
def test_metformin_above_threshold():
    bio.setup_loguru()
    polymers = get_polymers_above_threshold("metformin", threshold=50)
    config = Config(
        molecule_name="metformin",
        polymer_smiles=polymers,
    )
    explain_capacity_prediction(config)


def get_polymers_above_threshold(
    molecule_name: str, threshold: float = 50
) -> List[str]:
    """Read polymers above threshold from interesting_result CSV."""
    csv_path = (
        RESULTS_DIR
        / "interesting_result"
        / molecule_name.lower()
        / "04_predicted_capacities.csv"
    )
    df = pd.read_csv(csv_path)
    return df[df["PREDICTED_CAPACITY"] > threshold]["POLYMER_USED"].tolist()


def explain_capacity_prediction(config: Config):
    """Explain capacity predictions for multiple polymers using SHAP."""
    assert config.save_dir, "save_dir must be set in config"
    save_dir = config.save_dir / config.molecule_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_yaml_path = save_dir / "config.yaml"
    yaml.SafeDumper.add_multi_representer(
        Path, lambda dumper, data: dumper.represent_str(str(data))
    )
    formatted_config = yaml.safe_dump(
        asdict(config), default_flow_style=False, sort_keys=False
    )
    with open(config_yaml_path, "w") as f:
        f.write(formatted_config)
    logger.info(f"Saved run configuration to {config_yaml_path.name}")

    logger.info(
        f"Analyzing {len(config.polymer_smiles)} polymer(s) for molecule: {config.molecule_name}"
    )
    logger.info("Fetching drug SMILES...")
    drug_smiles = bio.get_smiles_from_name(config.molecule_name)
    if not drug_smiles:
        logger.error(f"Could not find SMILES for molecule: {config.molecule_name}")
        return
    logger.info(f"Drug SMILES: {drug_smiles}")

    logger.info("Loading trained model...")
    pscp = PeeSmileCapacityPredictor()
    model = pscp.load_trained_model()
    model.eval()

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

    logger.info("Computing background features...")
    background_features = model.featurize_fn(background_sample)
    feature_columns = [
        col for col in background_features.columns if col not in ["CAPACITY", "SOURCE"]
    ]
    expected_n_features = (
        model.x_scaler.n_features_in_ if model.x_scaler is not None else None
    )
    if expected_n_features and len(feature_columns) > expected_n_features:
        feature_columns = feature_columns[:expected_n_features]
        logger.warning(f"Truncated to {expected_n_features} features")
    elif expected_n_features and len(feature_columns) < expected_n_features:
        logger.warning(
            f"Only have {len(feature_columns)} features, expected {expected_n_features}"
        )

    logger.info(f"Using {len(feature_columns)} features for SHAP analysis")

    background_numeric = (
        background_features[feature_columns]
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

    explainer = shap.Explainer(model_predict, background_numeric, silent=True)

    all_shap_values = []
    all_predictions = []
    all_features = []

    for i, polymer_smiles in enumerate(config.polymer_smiles):
        logger.info(
            f"Processing polymer {i + 1}/{len(config.polymer_smiles)}: {polymer_smiles[:50]}..."
        )

        polymer_df = pd.DataFrame(
            {
                "POLYMER_USED": [polymer_smiles],
                "DRUG": [drug_smiles],
                "WATER_PH": [config.water_ph],
                "CONCENTRATION": [config.concentration],
            }
        )

        prediction = model.predict(polymer_df)
        logger.info(f"  Predicted capacity: {prediction:.2f}")
        all_predictions.append(prediction)

        polymer_features = model.featurize_fn(polymer_df)
        common_cols = [
            col for col in feature_columns if col in polymer_features.columns
        ]
        if len(common_cols) < len(feature_columns):
            logger.warning(
                f"Missing {len(feature_columns) - len(common_cols)} columns for polymer {i + 1}"
            )

        polymer_numeric = (
            polymer_features[common_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.astype(float)
        )

        shap_values = explainer.shap_values(polymer_numeric)
        all_shap_values.append(shap_values)
        all_features.append(polymer_features)

    logger.info("PREDICTION SUMMARY:")
    for i, (polymer, pred) in enumerate(zip(config.polymer_smiles, all_predictions)):
        logger.info(f"Polymer {i + 1}: {pred:.2f}")

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_shap = np.vstack(
        [
            np.array(sv).flatten() if np.array(sv).ndim == 1 else np.array(sv)[0]
            for sv in all_shap_values
        ]
    )

    plt.figure(figsize=(14, 10))
    shap.summary_plot(combined_shap, combined_features, show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=300)
    plt.close()
    logger.info(f"Saved shap_summary.png")

    plt.figure(figsize=(14, 10))
    shap.summary_plot(combined_shap, combined_features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary_bar.png", dpi=300)
    plt.close()
    logger.info(f"Saved shap_summary_bar.png")

    feature_importance_mean = np.abs(combined_shap).mean(axis=0)
    top_3_indices = np.argsort(feature_importance_mean)[-3:][::-1]
    top_3_feature_names = [combined_features.columns[i] for i in top_3_indices]

    for idx, feature_name in enumerate(top_3_indices):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            combined_shap,
            combined_features,
            show=False,
            interaction_index="auto",
        )
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_dependence_{idx + 1}_{feature_name}.png", dpi=300)
        plt.close()
        logger.info(f"Saved shap_dependence_{idx + 1}_{feature_name}.png")

    baseline_value = model_predict(background_numeric).mean()

    plt.figure(figsize=(14, 10))
    shap.decision_plot(
        baseline_value,
        combined_shap,
        combined_features,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "shap_decision.png", dpi=300)
    plt.close()
    logger.info(f"Saved shap_decision.png")

    for i, (polymer, shap_vals) in enumerate(
        zip(config.polymer_smiles, all_shap_values)
    ):
        shap_arr = np.array(shap_vals)
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.flatten()
        elif shap_arr.ndim > 1:
            shap_arr = shap_arr[0]

        plt.figure(figsize=(12, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_arr,
                base_values=baseline_value,
                data=all_features[i].iloc[0].values,
                feature_names=all_features[i].columns.tolist(),
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_waterfall_polymer_{i + 1}.png", dpi=300)
        plt.close()
        logger.info(f"Saved shap_waterfall_polymer_{i + 1}.png")

    top_features_data = []
    for i, (polymer, shap_vals) in enumerate(
        zip(config.polymer_smiles, all_shap_values)
    ):
        shap_arr = np.array(shap_vals)
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.flatten()
        elif shap_arr.ndim > 1:
            shap_arr = shap_arr[0]
        feature_importance = pd.DataFrame(
            {
                "polymer_idx": i + 1,
                "feature": all_features[i].columns,
                "shap_value": shap_arr,
            }
        ).sort_values("shap_value", ascending=False)
        top_features_data.append(feature_importance)

    all_top_features = pd.concat(top_features_data, ignore_index=True)
    all_top_features.to_csv(save_dir / "top_features.csv", index=False)
    logger.info(f"Saved top_features.csv")

    logger.info("TOP 10 FEATURES FOR EACH POLYMER:")
    for i, df in enumerate(top_features_data):
        logger.info(f"\n--- Polymer {i + 1} ---\n{df.head(10).to_string(index=False)}")

    polymer_comparison = pd.DataFrame(
        {
            "polymer_idx": list(range(1, len(config.polymer_smiles) + 1)),
            "polymer_smiles": config.polymer_smiles,
            "predicted_capacity": all_predictions,
        }
    )

    for i, df in enumerate(top_features_data):
        top5_features = df.head(5)["feature"].tolist()
        polymer_comparison.loc[i, "top_5_features"] = "; ".join(top5_features)

    polymer_comparison.to_csv(save_dir / "polymer_comparison.csv", index=False)
    logger.info(f"Saved polymer_comparison.csv")

    logger.info(f"POLYMER COMPARISON:\n{polymer_comparison.to_string(index=False)}")
    logger.info(f"All outputs saved to: {save_dir}")
