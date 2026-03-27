import torch
import math
import pandas as pd
from typing import Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
import lele, bio
from sklearn.preprocessing import StandardScaler
from bio.Dataset import PDCCMethod
from loguru import logger

"""
PDCC — Polymer Drug Concentration Capacity
"""

@dataclass
class Config:
    csv_file: Path
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_size: Optional[int] = None
    seed: int = 42
    
    
@dataclass
class PDCC:
    df: pd.DataFrame
    config: Config
    
    def __init__(self, config: Config):
        self.df = pd.read_csv(config.csv_file)
        self.config = config
    
    @staticmethod
    def featurize_fn(df: pd.DataFrame) -> pd.DataFrame:
        return PDCCMethod.featurize(df)
        
    def increment_dataset(self, options = PDCCMethod.increment_dataset.Options()):
        orginal_len = len(self.df)
        self.df = PDCCMethod.increment_dataset(self.df, options)
        logger.info(f"Incremented dataset: gained {len(self.df) - orginal_len} data.")
        
        
    def convert_names_to_smiles(self, options = PDCCMethod.convert_names_to_smiles.Options()):
        self.df = PDCCMethod.convert_names_to_smiles(self.df, options)
        
    def to_torch_dataset(self) -> torch.utils.data.Dataset:
        return bio.Dataset.TorchDataset.PDCCtorch(self)


def _count_origins(df: pd.DataFrame) -> int:
    origins_mask = (df['CONCENTRATION'] == 0.0) & (df['CAPACITY'] == 0.0)
    origins_df = df[oarigins_mask]
    return len(origins_df)


def test_usage():
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    config = Config(csv_file=PDCC_CSV)
    dataset = PDCC(config)
    dataset.increment_dataset(
        options = PDCCMethod.increment_dataset.Options(
            method="interpolate_then_add_origins",
            n_points=2,
        )
    )
    dataset.convert_names_to_smiles()
    logger.info(f"Found {_count_origins(dataset.df)} origin points in the DataFrame.")
    
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
