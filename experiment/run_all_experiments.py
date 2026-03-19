import yaml, json
import tyro
from typing import Optional
from dataclasses import asdict
from itertools import product
from pathlib import Path
import lele, bio
from experiment.__call__ import Experiment, DatasetConfig, ModelConfig, FeaturizerConfig, run
from loguru import logger

FINGERPRINT_FEATURIZER = FeaturizerConfig(
    capping_atoms = ['H'],
    molecule_features_to_calculate = ['fingerprint'],
    polymer_features_to_calculate = ['fingerprint'],
    molecule_polymetrix_features = [],
    polymer_polymetrix_features = [],
    sidechain_polymetrix_features = [],
    backbone_polymetrix_features = [],
)
POLYMETRIX_FEATURIZER = FeaturizerConfig(
    capping_atoms = ['H'],
    molecule_features_to_calculate = [],
    polymer_features_to_calculate = [],
    molecule_polymetrix_features = ['ALL'],
    polymer_polymetrix_features = ['ALL'],
    sidechain_polymetrix_features = ['ALL'],
    backbone_polymetrix_features = ['ALL'],
)
MIXED_FEATURIZER = FeaturizerConfig(
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

FORCE_RERUN = False
EXPERIMENTS = [
    [Experiment(
        name="experiment_baseline",
    ), FORCE_RERUN],
    [Experiment(
        name="experiment_small_data",
        dataset=DatasetConfig(max_size=10),
        model=ModelConfig(epochs=100),
    ), FORCE_RERUN],
    [Experiment(
        name="experiment_mae_loss",
        model=ModelConfig(criterion="mae")
    ), FORCE_RERUN],
    [Experiment(
        name = "experiment_loocv_only_fingerprints_big",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
        ),
        features = FINGERPRINT_FEATURIZER,
    ), FORCE_RERUN],
    [Experiment(
        name = "experiment_loocv_only_fingerprints_small",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = FINGERPRINT_FEATURIZER,
    ), FORCE_RERUN],
    [Experiment(
        name = "experiment_loocv_only_polymetrix_big",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = POLYMETRIX_FEATURIZER,
    ), FORCE_RERUN],
    [Experiment(
        name = "experiment_loocv_only_polymetrix_small",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = POLYMETRIX_FEATURIZER,
    ), FORCE_RERUN],
    [Experiment(
        name = "experiment_loocv_mixed_features_big",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = MIXED_FEATURIZER,
    ), FORCE_RERUN],
    [Experiment(
        name = "experiment_loocv_mixed_features_small",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = MIXED_FEATURIZER,
    ), FORCE_RERUN],
]
ALL_EXPERIMENTS = EXPERIMENTS


# grid_lrs = [
#     1e-3, 5e-4
# ]
# grid_dropouts = [
#     0.1, 0.3
# ]
# grid_hidden_dims = [
#     [128, 64, 32], 
#     [256, 128, 64]
# ]
# GRID_EXPERIMENTS_1 = [
#     [Experiment(
#         name=f"exp_lr{lr}_drop{drop}_hd{len(hd)} ({str(hd)})",
#         model=ModelConfig(learning_rate=lr, dropout=drop, hidden_dims=hd)
#     ), FORCE_RERUN]
#     for lr, drop, hd in product(grid_lrs, grid_dropouts, grid_hidden_dims)
# ]
# ALL_EXPERIMENTS += GRID_EXPERIMENTS_1


# grid_k_folds = [
#     5, 10, 25, -1 # -1 means leave-one-out cross-validation
# ]
# grid_criterions = [
#     "mae", "mse"
# ]
# grid_hidden_dims = [
#     [128, 64, 32], 
#     [256, 128, 64],
#     [64, 32, 16],
#     [32, 16, 8],
#     [8, 8, 8, 4],
# ]
# GRID_EXPERIMENTS_2 = [
#     [Experiment(
#         name=f"exp_{k_fold}_fold_{criterion}_hd{len(hd)} ({str(hd)})",
#         model=ModelConfig(k_fold=k_fold, criterion=criterion, hidden_dims=hd)
#     ), FORCE_RERUN]
#     for k_fold, criterion, hd in product(grid_k_folds, grid_criterions, grid_hidden_dims)
# ]
# ALL_EXPERIMENTS += GRID_EXPERIMENTS_2



# grid_k_folds = [
#     5, 10, 25, -1 # -1 means leave-one-out cross-validation
# ]
# grid_hidden_dims = [
#     [128, 64, 32], 
#     [256, 128, 64],
#     [64, 32, 16],
#     [32, 16, 8],
#     [8, 8, 8, 4],
# ]
# GRID_EXPERIMENTS_3 = [
#     [Experiment(
#         name=f"exp_{feat_name}_{k_fold}_fold_hd{len(hd)} ({str(hd)})",
#         model=ModelConfig(k_fold=k_fold, hidden_dims=hd),
#         features=feat_config,
#     ), FORCE_RERUN]
#     for (feat_name, feat_config), k_fold, hd in product(
#         FEATURIZERS.items(), grid_k_folds, grid_hidden_dims
#     )
# ]
# ALL_EXPERIMENTS += GRID_EXPERIMENTS_3

def test_(): 
    run_all_experiments()

def run_all_experiments():
    save_dir = lele.P(__file__).parent  
    
    for exp, force_rerun in ALL_EXPERIMENTS:
        logger.info(f"Evaluating {exp.name}...")
        
        if not force_rerun and is_experiment_unchanged(exp, save_dir):
            logger.info(f"⏭️ Skipping {exp.name} - Configuration is unchanged.")
            continue
            
        logger.info(f"🚀 Starting {exp.name}...")
        try:
            run(exp)
            logger.info(f"✅ Successfully finished {exp.name}.")
        except Exception as e:
            logger.error(f"❌ Failed {exp.name} with error: {e}")


def _normalize_config(config_dict: dict) -> dict:
    """Converts a dict with complex objects (like Path) into native strings/types."""
    # A quick trick: dumping to JSON with default=str converts Paths to strings,
    # then loading it back gives us a clean dictionary comparable to parsed YAML.
    return json.loads(json.dumps(config_dict, default=str))

def is_experiment_unchanged(experiment: Experiment, save_dir: Path) -> bool:
    """Checks if the experiment has already been run with the exact same config."""
    config_path = save_dir / experiment.name / "config.yaml"
    
    if not config_path.exists():
        return False
        
    try:
        with open(config_path, "r") as f:
            saved_config = yaml.safe_load(f)
            
        current_config = _normalize_config(asdict(experiment))
        
        return saved_config == current_config
    except Exception as e:
        logger.warning(f"Could not read/compare config for {experiment.name} ({e}). Defaulting to rerun.")
        return False
