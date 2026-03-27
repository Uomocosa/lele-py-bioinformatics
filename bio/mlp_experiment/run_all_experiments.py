import yaml, json
import tyro
from typing import Optional
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lele, bio
from bio.__global__ import RESULTS_DIR
from loguru import logger

ExperimentConfig = bio.mlp_experiment.Config
DatasetConfig = bio.mlp_experiment.Config.DatasetConfig
ModelConfig = bio.mlp_experiment.Config.ModelConfig
FeaturizerOptions = bio.mlp_experiment.Config.FeaturizerOptions

    
def test_run_all_experiments(): 
    # pixi run pytest -rFP -q -s bio\mlp_experiment\run_all_experiments.py::test_run_all_experiments -o "addopts="
    rank_experiments()


def test_rank_all_experiments():
    # pixi run pytest -rFP -q -s bio\mlp_experiment\run_all_experiments.py::test_rank_all_experiments -o "addopts="
    rank_experiments()

FINGERPRINT_FEATURIZER = FeaturizerOptions(
    capping_atoms = ['H'],
    molecule_features_to_calculate = ['fingerprint'],
    polymer_features_to_calculate = ['fingerprint'],
    molecule_polymetrix_features = [],
    polymer_polymetrix_features = [],
    sidechain_polymetrix_features = [],
    backbone_polymetrix_features = [],
)
POLYMETRIX_FEATURIZER = FeaturizerOptions(
    capping_atoms = ['H'],
    molecule_features_to_calculate = [],
    polymer_features_to_calculate = [],
    molecule_polymetrix_features = ['ALL'],
    polymer_polymetrix_features = ['ALL'],
    sidechain_polymetrix_features = ['ALL'],
    backbone_polymetrix_features = ['ALL'],
)
MIXED_FEATURIZER = FeaturizerOptions(
    capping_atoms = ['H'],
    molecule_features_to_calculate = ['fingerprint'],
    polymer_features_to_calculate = [],
    molecule_polymetrix_features = [],
    polymer_polymetrix_features = ['ALL'],
    sidechain_polymetrix_features = ['ALL'],
    backbone_polymetrix_features = ['ALL'],
)
FEATURIZERS_DICT = {
    'only_fingerprints': FINGERPRINT_FEATURIZER,
    'only_polymetrix': POLYMETRIX_FEATURIZER,
    'mixed_features': MIXED_FEATURIZER,
}

""" 
Note! The first 4 experiments were run, but they are missing the
      "DEFUALT" increment dataset options. I dont want to run them
      again, it takes too long.
""" 
EXPERIMENTS = [
    ExperimentConfig(
        name="experiment_default",
    ),
    ExperimentConfig(
        name="experiment_hd_32_32_32",
        model_config=ModelConfig(
            hidden_dims = [32, 32, 32],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_16_16",
        model_config=ModelConfig(
            hidden_dims = [16, 16, 16],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_8_8_8_8",
        model_config=ModelConfig(
            hidden_dims = [8, 8, 8, 8],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_8_8",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 8, 8],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_32_16_8_8",
        model_config=ModelConfig(
            hidden_dims = [32, 16, 8, 8],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_32_16_8_4",
        model_config=ModelConfig(
            hidden_dims = [32, 16, 8, 4],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_8_8_8_8_8",
        model_config=ModelConfig(
            hidden_dims = [8, 8, 8, 8, 8],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_mae",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
            criterion_fn="mae",
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_basic_forward_fn",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
        ),
        forward_fn="basic"
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_4",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4, 4],
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_4_mae",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4, 4],
            criterion_fn="mae",
        ),
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_only_x_scaler",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
        ),
        y_scaler_fn = None,
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_only_y_scaler",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
        ),
        x_scaler_fn = None,
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_no_scaler",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
        ),
        x_scaler_fn = None,
        y_scaler_fn = None,
    ),
    ExperimentConfig(
        name="experiment_hd_16_8_4_4_4_no_scaler_basic_forward_fn",
        model_config=ModelConfig(
            hidden_dims = [16, 8, 4, 4, 4],
        ),
        x_scaler_fn = None,
        y_scaler_fn = None,
        forward_fn="basic",
    ),
]



def run_all_experiments():
    for exp_config in EXPERIMENTS:
        bio.mlp_experiment.run_with_config(exp_config)
        
        
def rank_experiments(base_dir: Path = RESULTS_DIR / "mlp_experiments"):
    """
    Reads the individual fold predictions for LOOCV by scanning the file system,
    aggregates them, and calculates global metrics to rank the experiments.
    """
    results = []
    
    if not base_dir.exists():
        logger.error(f"Base directory '{base_dir}' does not exist.")
        return

    # Iterate through all experiment folders in the base directory
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        predictions_file = exp_dir / "fold_predictions.jsonl"
        config_file = exp_dir / "exp_config.yaml"
        
        # Check if predictions exist
        if not predictions_file.exists():
            logger.warning(f"No predictions found for experiment '{exp_name}'. Skipping.")
            continue
            
        # Optional but recommended: Check the saved config to ensure it's an LOOCV run
        if config_file.exists():
            with open(config_file, "r") as f:
                try:
                    config_data = yaml.safe_load(f)
                    # If it's not LOOCV, skip it to match your original logic
                    if config_data.get("k_fold", -1) != -1:
                        logger.warning(f"Experiment '{exp_name}' is not LOOCV. Skipping.")
                        continue
                except yaml.YAMLError:
                    logger.error(f"Could not parse config for '{exp_name}'.")
        
        # Read the JSON lines
        try:
            df_preds = pd.read_json(predictions_file, lines=True)
        except ValueError:
            logger.error(f"Failed to parse JSONL for '{exp_name}'.")
            continue
            
        if df_preds.empty: 
            continue
            
        y_true = np.array(df_preds["actual"].tolist())
        y_pred = np.array(df_preds["predicted"].tolist())
        
        results.append({
            "Experiment": exp_name,
            "Validation": "LOOCV",
            "Q2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        })
    
    df_results = pd.DataFrame(results)
    if df_results.empty:
        logger.info("No valid experiment results found to rank.")
        return
        
    # Sort the dataframe: Q2 is higher-is-better
    df_results = df_results.sort_values(by="Q2", ascending=False)
    df_results.reset_index(drop=True, inplace=True)
    
    logger.info(f"--- LOOCV Leaderboard (Ranked by Q2) ---")
    print(df_results.to_markdown())
