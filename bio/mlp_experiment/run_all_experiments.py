import yaml, json
import tyro
from typing import Optional
from dataclasses import asdict
from itertools import product
from pathlib import Path
import lele, bio
from loguru import logger

ExperimentConfig = bio.mlp_experiment.Config
DatasetConfig = bio.mlp_experiment.Config.DatasetConfig
ModelConfig = bio.mlp_experiment.Config.ModelConfig
FeaturizerOptions = bio.mlp_experiment.Config.FeaturizerOptions

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

EXPERIMENTS = [
    ExperimentConfig(
        name="experiment_default",
    ),
    ExperimentConfig(
        name="experiment_mae_loss",
        model=ModelConfig(criterion_fn="mae")
    ),
    ExperimentConfig(
        name = "experiment_loocv_only_fingerprints_big",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
        ),
        features = FINGERPRINT_FEATURIZER,
    ),
    ExperimentConfig(
        name = "experiment_loocv_only_fingerprints_small",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = FINGERPRINT_FEATURIZER,
    ),
    ExperimentConfig(
        name = "experiment_loocv_only_polymetrix_big",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = POLYMETRIX_FEATURIZER,
    ),
    ExperimentConfig(
        name = "experiment_loocv_only_polymetrix_small",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = POLYMETRIX_FEATURIZER,
    ),
    ExperimentConfig(
        name = "experiment_loocv_mixed_features_big",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = MIXED_FEATURIZER,
    ),
    ExperimentConfig(
        name = "experiment_loocv_mixed_features_small",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
        features = MIXED_FEATURIZER,
    ),
    ExperimentConfig(
        name = "experiment_loocv_all_features_256_128_64",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[256, 128, 64],
            epochs = 100, 
            early_stop_patience = 50,
        ),
    ),
    ExperimentConfig(
        name = "experiment_loocv_all_features_8_8_8_8_8",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 8, 8], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
    ),
    ExperimentConfig(
        name = "experiment_loocv_all_features_8_8_4_4_4_4",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 4, 4, 4, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
    ),
    ExperimentConfig(
        name = "experiment_loocv_all_features_8_8_8_4",
        model = ModelConfig(
            k_fold=-1, 
            hidden_dims=[8, 8, 8, 4], 
            epochs = 100, 
            early_stop_patience = 50,
        ),
    ),
]



def run_all_experiments():
    for exp_config in EXPERIMENTS:
        bio.mlp_experiment.run_with_config(exp_config)
        
        
def test_(): 
    run_all_experiments()
