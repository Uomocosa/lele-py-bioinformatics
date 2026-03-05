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
from sklearn.preprocessing import StandardScaler
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
    criterion: nn.Module = nn.MSELoss()
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    early_stop_patience: int = 1000
    best_model_save_dir: Optional[Path] = MLP_TEST_DIR
    seed: int = 42
    

class MLP(nn.Module):
    def __init__(
        self, 
        splitted_dataset: bio.Dataset.Splitted,
        featurize: Optional[Callable[pd.DataFrame, pd.DataFrame]],
        scaler: Optional[StandardScaler] = None,
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
        self.scaler = scaler
        
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
        
    def scale(self, x_tensor: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            x_np = x_tensor.numpy().reshape(1, -1)
            x_scaled = self.scaler.transform(x_np)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        return x_tensor
    
    def predict(self, *args, **kwargs) -> float:
        """Predicts capacity from raw inputs."""
        self.eval()
        device = next(self.parameters()).device
        df_features = self.featurize(*args, **kwargs)
        logger.debug(f"df_features:\n{df_features}")
        logger.debug(f"df_features.values:\n{df_features.values}")
        x_tensor = torch.tensor(df_features.values.astype(float), dtype=torch.float32)
        x_tensor = self.scale(x_tensor)
        x_tensor = x_tensor.to(device)
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        with torch.no_grad(): prediction = self.forward(x_tensor)
        return prediction.item()


def train_model(model: MLP):
    torch.manual_seed(model.config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model.to(device)
    train_ds = model.data.train
    val_ds = model.data.validation
    logger.info(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=model.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=model.config.batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=model.config.learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(model.config.epochs):
        model.train() 
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = model.config.criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_ds)

        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = model.config.criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_ds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        if model.config.early_stop_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= model.config.early_stop_patience:
                    logger.warning(f"EARLY STOPPING: Validation loss hasn't improved for {model.config.early_stop_patience} epochs.")
                    break 

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        logger.info(f"Restored best model weights with Val MSE: {best_val_loss:.4f}")

    logger.success("Training complete!")
    return model
    

def save_model(model: MLP):
    dir = model.config.best_model_save_dir
    if not dir: warnings.warn("Directory not specified."); return
    dir.mkdir(parents=True, exist_ok=True)
    unique_dir = model.config.best_model_save_dir / lele.String.unique()
    logger.debug(f"unique_dir: {unique_dir}")
    unique_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), unique_dir / "model.pt")
    config_dict = bio.Dataset.serialize_dataclass_instance(model.config)
    lele.Json.save_dict_to_jsonc_file(config_dict, unique_dir / "model_config.jsonc")
    

def check_model_accuracy(model: MLP) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_ds = model.data.test
    if not test_ds or len(test_ds) == 0:
        logger.warning("Test dataset is empty or not found.")
        return

    logger.info(f"Evaluating on Test Dataset (Size: {len(test_ds)})")
    test_loader = DataLoader(test_ds, batch_size=model.config.batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []

    # Disable gradient calculation for faster, memory-efficient inference
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            
            all_preds.append(predictions)
            all_targets.append(batch_y)

    # Combine all batches into single 1D tensors
    preds_tensor = torch.cat(all_preds).squeeze()
    targets_tensor = torch.cat(all_targets).squeeze()

    # Calculate Regression Metrics
    mse = F.mse_loss(preds_tensor, targets_tensor).item()
    mae = F.l1_loss(preds_tensor, targets_tensor).item()
    rmse = mse ** 0.5
    
    # Calculate R-squared (Coefficient of Determination)
    target_mean = torch.mean(targets_tensor)
    ss_tot = torch.sum((targets_tensor - target_mean) ** 2)
    ss_res = torch.sum((targets_tensor - preds_tensor) ** 2)
    r2 = (1 - (ss_res / ss_tot)).item()

    # Log the results
    logger.info("=== Test Set Evaluation Results ===")
    logger.info(f"MSE  (Mean Squared Error)      : {mse:.4f}")
    logger.info(f"RMSE (Root Mean Squared Error) : {rmse:.4f}")
    logger.info(f"MAE  (Mean Absolute Error)     : {mae:.4f}")
    logger.info(f"R² Score                       : {r2:.4f}")
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}



def test_():
    lele.Loguru.simple_format()
    dataset_config = bio.Dataset.PDCC.Config()
    model_config = Config(
        hidden_dims = [128, 64, 32],
        dropout = 0.2,
        epochs = 10,
        batch_size = 8,
        learning_rate = 1e-3,
        early_stop_patience = 1,
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
    scaler = splitted_dataset.scale(
        feature_col_indexes = range(torch_dataset.num_features),
        scaler_fn = StandardScaler()
    )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = dataset.featurize,
        scaler = scaler,
        config = model_config,
    )
    train_model(model)
    save_model(model)
    accuracy = check_model_accuracy(model)
    logger.info(f"model accuracy: {accuracy}")

    input_df = pd.DataFrame({
        'POLYMER_USED': ["*/CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O"],
        'DRUG': ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        'CONCENTRATION': ["12.5"],
    })
    
    prediction = model.predict(input_df)
    logger.info(f"Predicted Capacity: {prediction:.4f}")
