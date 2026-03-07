import sys
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field, replace
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from typing import Optional, Callable, Tuple
from polymetrix.featurizers.chemical_featurizer import *
from bio.Dataset import PDCC, PDCCMethod
from bio.ML import MLP, MLPMethod
import lele, bio
from bio.__global__ import BIOINFORMATICS_DIR, DATASETS_DIR
from bio.mlp_train import TRAIN_DATASET, VAL_DATASET, TEST_DATASET
from loguru import logger

ALL_FEATURES = [
    NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
    NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
    FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
    MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
    MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
    BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
]
# ALL polymetrix feature
# POLYMER_FEATURES = MOLECULE_FEATURES = SIDECHAIN_FEATURES = BACKBONE_FEATURES = ALL_FEATURES

# NO polymetrix feature
POLYMER_FEATURES = MOLECULE_FEATURES = SIDECHAIN_FEATURES = BACKBONE_FEATURES = []

DEFAULT_CAPPING_ATOMS = {
    'H': 1,
    # 'C': 6,
    # 'O': 8
}

@dataclass
class FeaturizeOptions(PDCCMethod.featurize.Options):
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS)
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048
    protonate_precision: float = 1.0
    molecule_features_to_calculate: list = field(default_factory=lambda: [
        # 'logp',
        # 'logd',
        # 'homo_lumo_eV',
        'fingerprint',
    ])
    polymer_features_to_calculate: list = field(default_factory=lambda: [
        # 'logp', 
        # 'logd', 
        # 'homo_lumo_eV',
        'fingerprint',
    ])
    molecule_polymetrix_features: list = field(default_factory=lambda: MOLECULE_FEATURES)
    polymer_polymetrix_features: list = field(default_factory=lambda: POLYMER_FEATURES)
    sidechain_polymetrix_features: list = field(default_factory=lambda: SIDECHAIN_FEATURES)
    backbone_polymetrix_features: list = field(default_factory=lambda: BACKBONE_FEATURES)
    
@dataclass
class ModelConfig(MLP.Config):
    k_fold: int = 5
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    criterion: nn.Module = nn.MSELoss()
    learning_rate: float = 1e-3
    epochs: int = 1000
    early_stop_patience: int = 100
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    best_model_save_dir: Optional[Path] = bio.ML.MLP.MLP_TEST_DIR
    
CPU_MODEL_CONFIG = ModelConfig(
    epochs = 500,
    early_stop_patience = 50,
    dropout = 0.3,
    hidden_dims = [1024, 256, 64],
)

CUDA_MODEL_CONFIG = ModelConfig(
    epochs = 10000,
    early_stop_patience = 1000,
    dropout = 0.3,
    hidden_dims = [2048, 1024, 256, 64, 8],
    num_workers = 4,
    batch_size = 256,
)

def main_cuda():
    # pixi run -e cuda python -c "from bio.mlp_train_featurize_k5_fold_only_fingerprints_2 import main_cuda; main_cuda()"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    dataset_config = bio.Dataset.PDCC.Config()
    featurize_options = FeaturizeOptions()
    featurize_fn = lambda df: PDCCMethod.featurize(df, featurize_options)
    run_with_config(
        dataset_config,
        CUDA_MODEL_CONFIG,
        featurize_fn,
    )

def test_():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    dataset_config = bio.Dataset.PDCC.Config(
        max_size = 10,
    )
    featurize_options = FeaturizeOptions(
        molecule_features_to_calculate = ['logd'],
        polymer_features_to_calculate = ['logd'],
        molecule_polymetrix_features = [],
        polymer_polymetrix_features = [],
        sidechain_polymetrix_features = [],
        backbone_polymetrix_features = [],
    )
    featurize_fn = lambda df: PDCCMethod.featurize(df, featurize_options)
    run_with_config(
        dataset_config,
        CPU_MODEL_CONFIG,
        featurize_fn,
    )
    # logger.add(sys.stderr, level="WARNING")


def run_with_config(
    dataset_config, 
    model_config, 
    featurize_fn, 
):
    assert dataset_config.seed == model_config.seed
    bio.ML.set_seed(dataset_config.seed)
    save_dir = model_config.best_model_save_dir
    assert save_dir is not None
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = bio.Dataset.PDCC(dataset_config)
    dataset.featurize = featurize_fn
    torch_dataset = dataset.to_torch_dataset()
    trn, val, tst = dataset_config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = dataset_config.seed,
    )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = featurize_fn,
        config = model_config,
        scaler = None,
    )
    
    MLPMethod.k_fold_cross_validation(model, k=model_config.k_fold)
