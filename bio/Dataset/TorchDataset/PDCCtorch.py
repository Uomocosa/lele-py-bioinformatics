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
        self.df = pd.read_csv(dataset.config.csv_file)
        if dataset.config.max_size: self.df = self.df.head(dataset.config.max_size)
        
        required_columns = ['POLYMER_USED', 'DRUG', 'CONCENTRATION', 'CAPACITY']
        for col in required_columns:
            if col in self.df.columns: continue
            raise ValueError(f"Dataset is missing required column: {col}\n Dataset has the following columns: {self.df.columns}")
        
        self.df = dataset.featurize(self.df)
        
        y = self.df['CAPACITY'].values
        X = self.df.drop(columns=['CAPACITY']).values
        
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
