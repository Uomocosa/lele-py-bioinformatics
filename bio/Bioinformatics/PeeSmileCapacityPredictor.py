import sys, types
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field, asdict
import copy, warnings
from pathlib import Path
from typing import Optional, Callable, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lele, bio
from bio.ML import MLP, MLPMethod
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import PDCC_CSV, RESULTS_DIR
from loguru import logger

SAVE_MODEL_DIR = RESULTS_DIR / "PeeSmileCapacityPredictor"

"""
Note! The defaults for PeeSmileCapacityPredictor were taken after the 
      best result obtained from 'bio.mlp_experiments.run_all_experiments'
"""

def train():
    pscp = PeeSmileCapacityPredictor()
    trained_model = pscp.get_trained_model()
    pscp.save_trained_model(trained_model)
    pscp.plot_learning_curve()
    trained_model = None
    trained_model = pscp.load_trained_model()
    assert trained_model is not None, "Failed to load trained model"


def test_train(): 
    train()

def test_usage():
    pscp = PeeSmileCapacityPredictor()
    trained_model = pscp.load_trained_model()
    assert trained_model is not None, "Failed to load trained model"
    
    trained_model.eval()
    input_df = pd.DataFrame({
        'POLYMER_USED': ["*/CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O"],
        'DRUG': ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        'WATER_PH': [6.5],
        'CONCENTRATION': [12.5],
    })
    prediction = trained_model.predict(input_df)
    logger.info(f"Predicted Capacity: {prediction:.4f}")

    


@dataclass
class DatasetConfig(PDCC.Config):
    csv_file: Path = PDCC_CSV
    train_validation_test_pecentages: Tuple[float, float, float] = (1, 0, 0)
    
@dataclass
class ModelConfig(MLP.Config):
    save_dir: Optional[Path] = SAVE_MODEL_DIR
    hidden_dims: list = field(default_factory=lambda: [16, 8, 4, 4, 4])
    epochs: int = 100
    
FeaturizerOptions = PDCCMethod.featurize.Options
IncrementDatasetOptions = PDCCMethod.increment_dataset.Options


SCALER_FN_MAP = {
    "standard": StandardScaler(),
    "min_max": MinMaxScaler(),
    "min_max_range01": MinMaxScaler(feature_range=(0, 1)),
}


def forward_fn(mlp, x):
    x = mlp.model(x)
    return mlp.output(x)
def forward_softplus_fn(mlp, x):
    x = mlp.model(x)
    return F.softplus(mlp.output(x))
FORWARD_FN_MAP = {
    "": forward_fn,
    "basic": forward_fn,
    "softplus": forward_softplus_fn,
}

@dataclass
class PeeSmileCapacityPredictor():
    x_scaler_fn: Optional[str] = "standard"
    y_scaler_fn: Optional[str] = "min_max_range01"
    save_dir: Optional[Path] = SAVE_MODEL_DIR
    forward_fn: str = "softplus"
    dataset_config: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())
    featurizer_options: FeaturizerOptions = field(default_factory=lambda: FeaturizerOptions())
    incerement_dataset_options: IncrementDatasetOptions = field(default_factory=lambda: IncrementDatasetOptions())
    seed: int = 42

    def get_x_scaler_fn(self):
        if not self.x_scaler_fn: return None
        assert self.x_scaler_fn in SCALER_FN_MAP, f"Unknown x_scaler_fn: {self.x_scaler_fn}, must be one of {list(SCALER_FN_MAP.keys())}"
        return SCALER_FN_MAP[self.x_scaler_fn]
        
    def get_y_scaler_fn(self):
        if not self.y_scaler_fn: return None
        assert self.y_scaler_fn in SCALER_FN_MAP, f"Unknown y_scaler_fn: {self.y_scaler_fn}, must be one of {list(SCALER_FN_MAP.keys())}"
        return SCALER_FN_MAP[self.y_scaler_fn]
        
    def get_forward_fn(self):
        assert self.forward_fn in FORWARD_FN_MAP, f"Unknown forward_fn: {self.forward_fn}, must be one of {list(FORWARD_FN_MAP.keys())}"
        return FORWARD_FN_MAP[self.forward_fn]
        
    def get_trained_model(self) -> MLP:
        return _get_trained_model(self)
        
    def save_trained_model(self, trained_model: MLP, filename: str = "weights_and_scalers.pt"):
        """Saves the trained model weights, scalers, and dimensions."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / filename
        
        checkpoint = {
            'model_weights': trained_model.state_dict(),
            'x_scaler': trained_model.x_scaler,
            'y_scaler': trained_model.y_scaler,
            'input_dim': trained_model.input_dim,
            'output_dim': getattr(trained_model, 'output_dim', 1) # Added output_dim for completeness
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Model explicitly saved to {save_path}")


    def load_trained_model(self, filename: str = "weights_and_scalers.pt") -> MLP:
        """Loads the model and scalers without reading the original CSV."""
        load_path = self.save_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}. Have you trained/saved it yet?")
            
        checkpoint = torch.load(load_path, weights_only=False)
        
        # 1. Duck-type a tiny fake dataset so MLP.__init__ can calculate dimensions safely
        class MockDataset:
            def __init__(self, in_dim, out_dim):
                self.train = [(torch.zeros(in_dim), torch.zeros(out_dim))]
                
        mock_data = MockDataset(checkpoint['input_dim'], checkpoint.get('output_dim', 1))
        
        # 2. Rebuild the featurize function
        featurize_fn = lambda df: PDCCMethod.featurize(df, options=self.featurizer_options)
        
        # 3. Instantiate the architecture cleanly
        model = MLP(
            splitted_dataset=mock_data,
            featurize_fn=featurize_fn,
            x_scaler=checkpoint['x_scaler'],
            y_scaler=checkpoint['y_scaler'],
            config=self.model_config
        )
        
        # 4. Restore the overridden forward function & weights
        model.forward = types.MethodType(self.get_forward_fn(), model)
        model.load_state_dict(checkpoint['model_weights'])
        model.eval()
        
        logger.info(f"Model loaded successfully from {load_path}")
        return model
        

    def plot_learning_curve(self):
        """Plots Training loss (and Validation if available) against the number of epochs."""
        log = _get_log_files(self)  # FIXED: use self, not pscp
        log_file = log["traing_epochs"]  # FIXED: use log dictionary directly
        
        if not log_file.exists(): 
            logger.error(f"Cannot plot: Log file not found at {log_file}")
            return

        df_epochs = pd.read_json(log_file, lines=True)
        if not df_epochs.empty:
            import matplotlib.pyplot as plt # Ensure plt is imported
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_epochs, x="epoch", y="train_loss", label="Train Loss", linewidth=2, color="blue")
            
            # Safely plot validation loss only if it contains real numbers
            if "val_loss" in df_epochs.columns and not df_epochs["val_loss"].isna().all():
                sns.lineplot(data=df_epochs, x="epoch", y="val_loss", label="Validation Loss", linewidth=2, color="orange")
            
            criterion_name = self.model_config.criterion_fn
            plt.title("PeeSmileCapacityPredictor - Learning Curve")
            plt.xlabel("Epoch")
            plt.ylabel(f"Loss ({criterion_name.upper()})")
            plt.grid(True, linestyle='--', alpha=0.7) # Added grid for readability
            plt.legend()
            
            img_path = self.save_dir / "plot_learning_curve.png"
            plt.tight_layout()
            plt.savefig(img_path, dpi=300)
            plt.close()
            logger.info(f"Saved learning curves plot to: {img_path}")
    
    

def _get_trained_model(pscp: PeeSmileCapacityPredictor) -> MLP:
    _setup_loguru(pscp) 

    if pscp.dataset_config.seed != pscp.seed: 
        logger.warning(f"Dataset seed {pscp.dataset_config.seed} does not match pscp seed {pscp.seed}, setting seed to {pscp.seed}")
        pscp.dataset_config.seed = pscp.seed
    if pscp.model_config.seed != pscp.seed: 
        logger.warning(f"ModelConfig seed {pscp.model_config.seed} does not match pscp seed {pscp.seed}, setting seed to {pscp.seed}")
        pscp.model_config.seed = pscp.seed
    
    x_scaler_fn = pscp.get_x_scaler_fn()
    y_scaler_fn = pscp.get_y_scaler_fn()
    forward_fn = pscp.get_forward_fn()
    bio.ML.set_seed(pscp.seed)
    
    featurize_fn = lambda df: PDCCMethod.featurize(df, options=pscp.featurizer_options)
    dataset = bio.Dataset.PDCC(config = pscp.dataset_config)
    dataset.increment_dataset(options=pscp.incerement_dataset_options)
    dataset.convert_names_to_smiles()
    dataset.featurize_fn = featurize_fn
    
    torch_dataset = dataset.to_torch_dataset()
    x_sample, y_sample = torch_dataset[0]
    trn, val, tst = pscp.dataset_config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = pscp.seed,
    )
    
    x_scaler = None
    y_scaler = None
    if x_scaler_fn:
        x_scaler = splitted_dataset.scale(
            feature_col_indexes = range(torch_dataset.num_features),
            feature_attribute = "X",
            scaler_fn = x_scaler_fn,
        )
    if y_scaler_fn:
        y_scaler = splitted_dataset.scale(
            feature_col_indexes = range(len(y_sample.shape)),
            feature_attribute = "y",
            scaler_fn = y_scaler_fn,
        )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize_fn = dataset.featurize_fn,
        x_scaler = x_scaler,
        y_scaler = y_scaler,
        config = pscp.model_config,
    )
    model.forward = types.MethodType(forward_fn, model)

    trained_model = MLPMethod.train_model(model)
    return trained_model


def _setup_loguru(pscp: PeeSmileCapacityPredictor):
    log = _get_log_files(pscp)
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            "bio.ML.MLPMethod.train_model": "WARNING",
        },
        level = "INFO"
    )
    logger.add(
        lele.Loguru.CleanJSONLSink(log["traing_epochs"]),
        filter=lambda record: record["extra"].get("log_type") == "epoch_trace",
        level="TRACE",
    )
    return logger
    
    
def _get_log_files(pscp: PeeSmileCapacityPredictor):
    """Returns paths for logging output without relying on external config names."""
    return {
        "traing_epochs": pscp.save_dir / "traing_epochs.jsonl",
        "fold_predictions": pscp.save_dir / "fold_predictions.jsonl",
        "fold_metrics": pscp.save_dir / "fold_metrics.jsonl",
        "aggregate": pscp.save_dir / "aggregated_results.jsonl",
    }
