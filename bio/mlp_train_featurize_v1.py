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

def test_():
    lele.Loguru.simple_format()
    dataset_config = bio.mlp_train.CommonDatasetConfig()
    model_config = bio.mlp_train.ModelConfig(
        hidden_dims = [128, 64, 32, 16, 8]
    )
    featurize = lambda df: PDCCMethod.featurize_v1(df)
    bio.mlp_train.run_with_config(
        dataset_config,
        model_config,
        featurize,
    )
