"""
TODO: needs a from_config function
TODO: remake the load function to be more similar to UnlabeledSmiles.load
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Iterator, Tuple
from dataclasses import dataclass
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
import lele, bio
from bio.Bioinformatics import Smile
from loguru import logger

SMILES_HEADER = 'SMILES' # 'SMILES' header in the csv file

@dataclass
class LabeledSmiles:
    config: bio.Dataset.Config
    smiles: List[Smile]
    features: torch.Tensor
    feature_names: List[str]
    tokenizer: Optional[SmilesTokenizer] = None
    
    def __len__(self) -> int:
        return len(self.smiles)
        
    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]: 
        return self.smiles[idx], self.features[idx]
        
    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:  
        return zip(self.smiles, self.features)
    
    def to_torch_dataset(self, block_size: int) -> torch.utils.data.Dataset:
        if self.tokenizer is None:
            from bio.Dataset.__global__ import SMILES_VOCABULARY
            logger.debug(f"SMILES_VOCABULARY @ {str(SMILES_VOCABULARY)}")
            self.tokenizer = SmilesTokenizer(str(SMILES_VOCABULARY))
        dataset = bio.Dataset.LabeledSmilesMethod.TorchDataset(
            parent=self, 
            tokenizer=self.tokenizer, 
            block_size=block_size, 
        )
        return dataset



def load(
    csv_file: Path,
    train_validation_test_pecentages = (0.6, 0.2, 0.2), 
    max_dataset_size = None
) -> LabeledSmiles:
    config = bio.Dataset.Config(csv_file, train_validation_test_pecentages, max_dataset_size)
    csv_file = lele.P(csv_file)
    assert csv_file.exists(), f"File '{csv_file}' does not exist"
    read_args = {}
    if max_dataset_size: read_args['nrows'] = max_dataset_size
    df = pd.read_csv(csv_file, **read_args)
    if SMILES_HEADER not in df.columns:
        raise ValueError(f"CSV must contain a '{SMILES_HEADER}' column")
    feature_cols = [c for c in df.columns if c != SMILES_HEADER]
    original_count = len(df)
    logger.debug("Canonicalizing SMILES...")
    canonicalize = bio.Bioinformatics.SmileMethod.canonicalize_smiles
    df['canonical'] = df[SMILES_HEADER].apply(canonicalize)
    df = df.dropna(subset=['canonical'])
    df = df.drop_duplicates(subset=['canonical'])
    clean_smiles = df['canonical'].tolist()
    clean_features = torch.tensor(
        df[feature_cols].values.astype(np.float32), 
        dtype=torch.float32
    )
    logger.info(f"Loaded: {len(clean_smiles)} (dropped {original_count - len(clean_smiles)})")
    logger.info(f"Features detected: {feature_cols}")
    return LabeledSmiles(
        config=config, 
        smiles=clean_smiles, 
        features=clean_features,
        feature_names=feature_cols
    )



def test_behaviour():
    from bio.Dataset.__global__ import ZINC_ZSCORE_CSV
    labled_smiles = load(ZINC_ZSCORE_CSV, max_dataset_size=100)
    print(f"First: {labled_smiles[0]}")
    print(f"Count: {len(labled_smiles)}")
    print(f"Source: {labled_smiles.config.csv_file}")
    

def test_to_torch_dataset():
    from bio.Dataset.__global__ import ZINC_ZSCORE_CSV
    labled_smiles = load(ZINC_ZSCORE_CSV, max_dataset_size=100)
    dataset = labled_smiles.to_torch_dataset(block_size=8)
    print(f"Example torch.Dataset: {dataset[0]}")
