import torch
import math
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
import bio
from bio.Dataset import PDCC
from loguru import logger

class PDCCtorch(Dataset):
    def __init__(self, dataset: PDCC):
        """
        Custom PyTorch Dataset for the Polymer Drug Concentration Capacity data.
        """
        # Load the dataset
        self.config = dataset.config
        self.df = dataset.df
        if dataset.config.max_size: self.df = self.df.head(dataset.config.max_size)
        
        required_columns = ['POLYMER_USED', 'DRUG', 'CONCENTRATION', 'CAPACITY']
        for col in required_columns:
            if col in self.df.columns: continue
            raise ValueError(f"Dataset is missing required column: {col}\n Dataset has the following columns: {self.df.columns}")
        
        self.metadata = self.df[['POLYMER_USED', 'DRUG']].copy()

        self.df = dataset.featurize_fn(self.df)

        # featurize() drops rows with NaN features (dropna preserves the original
        # index). Realign metadata to the surviving rows so metadata, X and y stay
        # row-for-row consistent — otherwise metadata.iloc[idx] (used for grouping
        # and per-prediction logging) points at the wrong polymer/drug.
        self.metadata = self.metadata.loc[self.df.index].reset_index(drop=True)

        y = self.df['CAPACITY'].values
        X = self.df.drop(columns=['CAPACITY']).select_dtypes(include='number').values
        
        # Convert to Tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def transform(self, scaler: StandardScaler):
        """Applies a fitted scaler to the feature tensor X."""
        # scaler.transform returns a numpy array; convert back to tensor
        scaled_X = scaler.transform(self.X.numpy()) 
        self.X = torch.tensor(scaled_X, dtype=torch.float32)
        return self

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    @property
    def num_features(self):
        """Helper property to dynamically pass input size to the MLP."""
        return self.X.shape[1]


def test_metadata_aligns_after_featurize_drops_rows():
    """metadata, X and y must stay row-for-row aligned even when featurize drops rows."""
    import pandas as pd
    from pathlib import Path
    from bio.Dataset import PDCC

    df = pd.DataFrame({
        'POLYMER_USED': ['pa', 'pb', 'pc'],
        'DRUG': ['da', 'db', 'dc'],
        'CONCENTRATION': [1.0, 2.0, 3.0],
        'CAPACITY': [10.0, 20.0, 30.0],
    })

    class _FakeDataset:
        config = PDCC.Config(csv_file=Path("dummy.csv"))
        def __init__(self, frame): self.df = frame
        def featurize_fn(self, d):
            # simulate featurize: drop name cols, drop a NaN row (index preserved), add a feature
            d = d.drop(columns=['POLYMER_USED', 'DRUG']).drop(index=1)
            d['feat'] = [0.5, 0.7]
            return d

    ds = PDCCtorch(_FakeDataset(df))
    assert len(ds.metadata) == len(ds) == ds.X.shape[0] == 2
    assert ds.metadata['POLYMER_USED'].tolist() == ['pa', 'pc']
    assert ds.metadata['DRUG'].tolist() == ['da', 'dc']
