import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Iterator, Tuple, Callable
from dataclasses import dataclass
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
import lele, bio
from bio.Bioinformatics import Smile
from bio.Dataset.__global__ import ZINC_BASE_CSV
from loguru import logger

@dataclass
class UnlabeledSmiles:
    config: bio.Dataset.Config
    smiles: pd.Series
    tokenizer: Optional[SmilesTokenizer] = None
    
    def __len__(self) -> int: return len(self.smiles)
    def __getitem__(self, idx) -> str: return self.smiles[idx]
    def __iter__(self) -> Iterator[str]:  return iter(self.smiles)
    
    def to_torch_dataset(self, block_size: int) -> torch.utils.data.Dataset:
        if self.tokenizer is None:
            from bio.Dataset.__global__ import SMILES_VOCABULARY
            self.tokenizer = SmilesTokenizer(SMILES_VOCABULARY)
        dataset = bio.Dataset.UnlabeledSmilesMethod.TorchDataset(
            self.smiles, 
            self.tokenizer, 
            block_size
        )
        return dataset


DatasetConfig = bio.Dataset.Config.Config
def from_config(config: DatasetConfig) -> UnlabeledSmiles:
    assert config.csv_file.exists(), f"File '{config.csv_file}' does not exist"
    read_args = {'header': None, 'names': ['smiles']} # Expects no header, only a long list of smiles
    if config.max_size: read_args['nrows'] = config.max_size
    df = pd.read_csv(config.csv_file, **read_args)
    original_count = len(df)
    smiles = df['smiles']
    if config.process_raw_smiles: smiles = config.process_raw_smiles(smiles)
    return UnlabeledSmiles(config=config, smiles=smiles)


def load(
    csv_file: Path = ZINC_BASE_CSV,
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    max_dataset_size: Optional[int] = None,
    process_raw_smiles: Optional[Callable[pd.Series, pd.Series]] = None,
) -> UnlabeledSmiles:
    dataset_config = DatasetConfig(
        csv_file = csv_file,
        train_validation_test_pecentages = train_validation_test_pecentages,
        max_size = max_dataset_size,
        process_raw_smiles = process_raw_smiles,
    )
    return from_config(dataset_config)
    
    
def test_process():
    from bio.Dataset.__global__ import ZINC_BASE_CSV
    
    def canonicalize(raw_smiles: pd.Series) -> pd.Series:
        logger.debug("Canonicalizing SMILES...")
        smiles = raw_smiles.apply(lambda s: Smile(s).canonicalize())
        smiles = smiles.dropna().drop_duplicates()
        return smiles
        
    config = bio.Dataset.Config(
        csv_file = ZINC_BASE_CSV,
        train_validation_test_pecentages = (0.6, 0.2, 0.2),
        max_size = 10,
        process_raw_smiles=canonicalize,
    )

    smiles_dataset = from_config(config)
    

def test_to_torch_dataset():
    from bio.Dataset.__global__ import ZINC_BASE_CSV
    config = bio.Dataset.Config(
        csv_file = ZINC_BASE_CSV,
        train_validation_test_pecentages = (0.6, 0.2, 0.2),
        max_size = 10,
    )
    smiles_dataset = from_config(config)
    smiles_dataset.to_torch_dataset(block_size=8)



def test_psmiles():
    from bio.Dataset.__global__ import PI1M
    config = bio.Dataset.Config(
        csv_file = PI1M,
        train_validation_test_pecentages = (0.6, 0.2, 0.2),
        max_size = 10,
    )
    smiles_dataset = from_config(config)
    smiles_dataset.to_torch_dataset(block_size=8)
