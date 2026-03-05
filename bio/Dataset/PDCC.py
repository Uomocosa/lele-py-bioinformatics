import torch
import math
import pandas as pd
from typing import Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
import lele, bio
from sklearn.preprocessing import StandardScaler
from bio.Dataset import PDCCMethod
from bio.__global__ import PDCC_DATASET, CONVERTED_PDCC_CSV
from loguru import logger

"""
PDCC — Polymer Drug Concentration Capacity
"""

@dataclass
class Config:
    csv_file: Path = CONVERTED_PDCC_CSV
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_size: Optional[int] = None
    seed: int = 42
    
    
@dataclass
class PDCC:
    config: Config
    
    def __init__(self, config: Config):
        self.config = config
    
    @staticmethod
    def featurize(df: pd.DataFrame):
        return PDCCMethod.featurize_v1(df)
        
    def to_torch_dataset(self) -> torch.utils.data.Dataset:
        return bio.Dataset.TorchDataset.PDCCtorch(self)


def test_usage():
    config = Config()
    dataset = PDCC(config)
    torch_dataset = dataset.to_torch_dataset()
    x_sample, y_sample = torch_dataset[0]
    logger.debug(f"Input features: {torch_dataset.num_features}")
    logger.debug(f"X shape: {x_sample.shape}") # Expect [num_features]
    logger.debug(f"y shape: {y_sample.shape}") # Expect [1]
    
    trn, val, tst = config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = config.seed,
    )
    logger.debug(f"splitted_dataset.train.dataset.X: {splitted_dataset.train.dataset.X}")
    
    scaler = splitted_dataset.scale(
        feature_col_indexes = range(torch_dataset.num_features),
        scaler_fn = StandardScaler()
    )
    logger.debug(f"scaler: {scaler}")
    logger.debug(f"splitted_dataset.train.dataset.X: {splitted_dataset.train.dataset.X}")
