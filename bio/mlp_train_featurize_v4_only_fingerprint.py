import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field, replace
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from typing import Optional, Callable, Tuple
from bio.Dataset import PDCC, PDCCMethod
from bio.ML import MLP
import lele, bio
from bio.__global__ import BIOINFORMATICS_DIR, DATASETS_DIR
from loguru import logger

DEFAULT_CAPPING_ATOMS = {
    "H": 1, 
    # "O": 8
}
@dataclass
class FeaturizeOptions(PDCCMethod.featurize_v4.Options):
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS)
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048


def test_():
    lele.Loguru.simple_format()
    dataset_config = bio.mlp_train.CommonDatasetConfig()
    model_config = bio.mlp_train.ModelConfig(
        hidden_dims = [
            2048, 
            1024, 
            512, 
            256, 
            128, 
            64, 
            32, 
            16, 
            8, 
            4,
        ],
        epochs = 10000,
        early_stop_patience = 1000,
    )
    featurize_options = FeaturizeOptions()
    featurize = lambda df: PDCCMethod.featurize_v4(df, featurize_options)
    bio.mlp_train.run_with_config(
        dataset_config,
        model_config,
        featurize,
    )
