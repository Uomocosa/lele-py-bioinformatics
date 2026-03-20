import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lele, bio
from bio.ML import MLP, MLPMethod
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import PDCC_CSV, RESULTS_DIR
from loguru import logger


SCALER_FN_MAP = {
    "standard": StandardScaler,
    "min_max": MinMaxScaler,
    "min_max_range01": lambda: MinMaxScaler(feature_range=(0, 1)),
}


def forward_fn(mlp, x):
    x = mlp.model(x)
    return mlp.output(x)
def forward_softplus_fn(mlp, x):
    x = mlp.model(x)
    return F.softplus(mlp.output(x))
FORWARD_FN_MAP = {
    "": forward_fn,
    "softplus": forward_softplus_fn,
}


@dataclass
class DatasetConfig(PDCC.Config):
    csv_file: Path = PDCC_CSV
    train_validation_test_pecentages: Tuple[float, float, float] = (1, 0, 0)
    
@dataclass
class ModelConfig(MLP.Config):
    save_dir: Optional[Path] = None
    
FeaturizerOptions = PDCCMethod.featurize.Options

@dataclass
class Config():
    name: str = "experiment_0"
    k_fold: int = -1 # Note! if you set it to -1 -> implies LOOCV otherwise k-fold cross validation
    save_dir: Path = RESULTS_DIR / "mlp_experiments"
    x_scaler_fn: str = "standard"
    y_scaler_fn: str = "min_max_range01"
    forward_fn: str = "softplus"
    seed: int = 42
    dataset_config: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())
    featurizer_options: FeaturizerOptions = field(default_factory=lambda: FeaturizerOptions())

    def get_x_scaler_fn(self):
        if not self.x_scaler_fn: return None
        assert self.x_scaler_fn in SCALER_FN_MAP, f"Unknown x_scaler_fn: {self.x_scaler_fn}, must be one of {list(SCALER_FN_MAP.keys())}"
        return SCALER_FN_MAP[self.x_scaler_fn]()
        
    def get_y_scaler_fn(self):
        if not self.y_scaler_fn: return None
        assert self.y_scaler_fn in SCALER_FN_MAP, f"Unknown y_scaler_fn: {self.y_scaler_fn}, must be one of {list(SCALER_FN_MAP.keys())}"
        return SCALER_FN_MAP[self.y_scaler_fn]()
        
    def get_forward_fn(self):
        assert self.forward_fn in FORWARD_FN_MAP, f"Unknown forward_fn: {self.forward_fn}, must be one of {list(FORWARD_FN_MAP.keys())}"
        return FORWARD_FN_MAP[self.forward_fn]



def test_():
    Config()
