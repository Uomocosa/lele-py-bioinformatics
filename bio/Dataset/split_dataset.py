import bio
import torch
from dataclasses import dataclass
from typing import Optional

def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_percentage: float,
    validation_percentage: float,
    test_percentage: float,
    seed: int,
) -> bio.Dataset.Splitted:
    total_pct = train_percentage + validation_percentage + test_percentage
    assert abs(total_pct - 1.0) < 1e-5, f"Percentages must sum to 1.0, got {total_pct}"
    
    total_len = len(dataset)
    train_len = int(train_percentage * total_len)
    val_len = int(validation_percentage * total_len)
    test_len = total_len - train_len - val_len # Give remainder to test to match total exactly
    
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, 
        [train_len, val_len, test_len],
        generator=generator
    )
    
    return bio.Dataset.Splitted(
        train=train_ds,
        validation=val_ds,
        test=test_ds
    )
    
    
def test_():
    from bio.Dataset.__global__ import ZINC_BASE_CSV
    import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)
    smiles = bio.Dataset.UnlabeledSmiles.load(ZINC_BASE_CSV, max_dataset_size=100)
    dataset = smiles.to_torch_dataset(block_size=8)
    splitted_dataset = split_dataset(
        dataset, 
        train_percentage=0.6,
        validation_percentage=0.2,
        test_percentage=0.2,
        seed=42
    )
    print(splitted_dataset)
