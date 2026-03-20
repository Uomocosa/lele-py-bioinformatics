import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from dataclasses import dataclass, field, asdict
import copy, warnings
import lele, bio
from pathlib import Path
from typing import Optional, Callable, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bio.__global__ import BIOINFORMATICS_DIR
from loguru import logger
SplittedDataset = bio.Dataset.Splitted.Splitted

"""
MLP — Multilayer Perceptron
"""

MLP_TEST_DIR = BIOINFORMATICS_DIR / "MLP_checkpoints_test"

@dataclass
class Config:
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    weight_decay: float = 1e-4
    criterion: nn.Module = nn.MSELoss()
    learning_rate: float = 1e-3
    epochs: int = 1000
    early_stop_patience: int = 100
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    best_model_save_dir: Optional[Path] = MLP_TEST_DIR
    

class MLP(nn.Module):
    def __init__(
        self, 
        splitted_dataset: bio.Dataset.Splitted,
        featurize: Optional[Callable[pd.DataFrame, pd.DataFrame]],
        x_scaler: Optional[StandardScaler] = None, 
        y_scaler: Optional[StandardScaler] = None,
        config: Config = Config()
    ):
        super(MLP, self).__init__()
        self.config = config
        
        sample_x, sample_y = splitted_dataset.train[0]
        self.input_dim = sample_x.shape[0]
        self.output_dim = sample_y.shape[0]
        logger.info(f"Inferred Architecture: Input({self.input_dim}) -> Hidden{config.hidden_dims} -> Output({self.output_dim})")
        self.data = splitted_dataset
        self.featurize = featurize
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        
        layers = []
        last_dim = self.input_dim
        
        # Build hidden layers dynamically
        for h_dim in config.hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(config.dropout))
            last_dim = h_dim
        
        # Output layer (Single value for CAPACITY)
        self.model = nn.Sequential(*layers)
        self.output = nn.Linear(last_dim, 1)
    
    def forward(self, x):
        x = self.model(x)
        return self.output(x)
        # return F.softplus(self.output(x))
        # return F.relu(self.output(x))
        
    def predict(self, *args, **kwargs) -> float:
        """Predicts capacity from raw inputs."""
        self.eval()
        device = next(self.parameters()).device
        df_features = self.featurize(*args, **kwargs)
        logger.debug(f"df_features:\n{df_features}")
        logger.debug(f"df_features.values:\n{df_features.values}")
        x_tensor = torch.tensor(df_features.values.astype(float), dtype=torch.float32)
        if self.x_scaler is not None:
            x_scaled = self.x_scaler.transform(x_tensor.numpy().reshape(1, -1))
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        x_tensor = x_tensor.to(device)
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        with torch.no_grad(): scaled_prediction = self.forward(x_tensor).cpu().numpy()
        if self.y_scaler is not None:
            real_prediction = self.y_scaler.inverse_transform(scaled_prediction)
        else:
            real_prediction = scaled_prediction
        return float(real_prediction.item())



def test_():
    lele.Loguru.simple_format()
    dataset_config = bio.Dataset.PDCC.Config()
    model_config = Config(
        hidden_dims = [8, 8, 8, 8],
        dropout = 0.2,
        epochs = 10,
        learning_rate = 1e-3,
        early_stop_patience = 1,
        batch_size = 8,
        seed = dataset_config.seed,
    )
    bio.ML.set_seed(dataset_config.seed)
    dataset = bio.Dataset.PDCC(dataset_config)
    torch_dataset = dataset.to_torch_dataset()
    x_sample, y_sample = torch_dataset[0]
    logger.debug(f"Input features: {torch_dataset.num_features}")
    logger.debug(f"X shape: {x_sample.shape}") # Expect [num_features]
    logger.debug(f"y shape: {y_sample.shape}") # Expect [1]
    
    trn, val, tst = dataset_config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = dataset_config.seed,
    )
    x_before, y_before = splitted_dataset.train[0]
    logger.debug(f"(BEFORE SCALING) Train[0] X: {x_before}")
    logger.debug(f"(BEFORE SCALING) Train[0] y: {y_before}")
    x_scaler = splitted_dataset.scale(
        feature_col_indexes = range(torch_dataset.num_features),
        scaler_fn = StandardScaler()
    )
    y_scaler = splitted_dataset.scale(
        feature_col_indexes = range(len(y_sample.shape)),
        feature_attribute = "y",
        scaler_fn = MinMaxScaler(feature_range=(0, 1))
    )
    x_after, y_after = splitted_dataset.train[0]
    logger.debug(f"(AFTER SCALING) Train[0] X: {x_after}")
    logger.debug(f"(AFTER SCALING) Train[0] y: {y_after}")
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = dataset.featurize,
        x_scaler = x_scaler,
        y_scaler = y_scaler,
        config = model_config,
    )
    def forward_fn(mlp, x):
        x = mlp.model(x)
        return F.softplus(mlp.output(x))
    model.forward = types.MethodType(forward_fn, model)
    bio.ML.MLPMethod.train_model(model)
    bio.ML.MLPMethod.save_model(model)
    accuracy = bio.ML.MLPMethod.check_model_accuracy(model)
    logger.info(f"model accuracy: {accuracy}")

    input_df = pd.DataFrame({
        'POLYMER_USED': ["*/CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O"],
        'DRUG': ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        'WATER_PH': ["6.5"],
        'CONCENTRATION': ["12.5"],
    })
    
    prediction = model.predict(input_df)
    logger.info(f"Predicted Capacity: {prediction:.4f}")
