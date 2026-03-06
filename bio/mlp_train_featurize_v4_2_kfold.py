import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field, replace
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from typing import Optional, Callable, Tuple
from bio.Dataset import PDCC, PDCCMethod
from bio.ML import MLP, MLPMethod
import lele, bio
from bio.__global__ import BIOINFORMATICS_DIR, DATASETS_DIR
from bio.mlp_train import TRAIN_DATASET, VAL_DATASET, TEST_DATASET
from loguru import logger

DEFAULT_CAPPING_ATOMS = {
    "H": 1, 
    # "O": 8
}
@dataclass
class FeaturizeOptions(PDCCMethod.featurize_v4.Options):
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS)
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048
    
    
CPU_MODEL_CONFIG = bio.mlp_train.ModelConfig(
    epochs = 500,
    early_stop_patience = 50,
    dropout = 0.3,
    learning_rate = 1e-3,
    hidden_dims = [
        # 2048, 
        1024, 
        # 512, 
        256, 
        # 128, 
        64, 
        # 32, 
        # 16, 
        # 8, 
        # 4,
        # 2,
    ],
)


def main():
    # pixi run -e cpu python -c "from bio.mlp_train_featurize_v4_2_kfold import main; main()"
    # pixi run -e cuda python -c "from bio.mlp_train_featurize_v4_2_kfold import main; main()"
    lele.Loguru.simple_format()
    dataset_config = bio.mlp_train.CommonDatasetConfig()
    featurize_options = FeaturizeOptions()
    featurize_fn = lambda df: PDCCMethod.featurize_v4(df, featurize_options)
    model_config = CPU_MODEL_CONFIG
    model_config.num_workers = 4
    model_config.batch_size = 128
    run_with_config(
        dataset_config,
        model_config,
        featurize_fn,
    )

def test_():
    # pixi run -e cpu python -c "from bio.mlp_train_featurize_v4_2_kfold import test_; test_()"
    # pixi run -e cuda python -c "from bio.mlp_train_featurize_v4_2_kfold import test_; test_()"
    lele.Loguru.simple_format()
    dataset_config = bio.mlp_train.CommonDatasetConfig()
    featurize_options = FeaturizeOptions()
    featurize_fn = lambda df: PDCCMethod.featurize_v4(df, featurize_options)
    run_with_config(
        dataset_config,
        CPU_MODEL_CONFIG,
        featurize_fn,
    )


def run_with_config(
    dataset_config, 
    model_config, 
    featurize_fn, 
):
    assert dataset_config.seed == model_config.seed
    bio.ML.set_seed(dataset_config.seed)
    save_dir = model_config.best_model_save_dir
    assert save_dir is not None
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_config = val_config = test_config = dataset_config
    train_config = replace(dataset_config, csv_file=TRAIN_DATASET)
    val_config = replace(dataset_config, csv_file=VAL_DATASET)
    test_config = replace(dataset_config, csv_file=TEST_DATASET)
    train_dataset = bio.Dataset.PDCC(train_config)
    val_dataset = bio.Dataset.PDCC(val_config)
    test_dataset = bio.Dataset.PDCC(test_config)
    train_dataset.featurize = featurize_fn
    val_dataset.featurize = featurize_fn
    test_dataset.featurize = featurize_fn
    train_torch_dataset = train_dataset.to_torch_dataset()
    val_torch_dataset = val_dataset.to_torch_dataset()
    test_torch_dataset = test_dataset.to_torch_dataset()
    
    splitted_dataset = bio.Dataset.Splitted.from_datasets(
        train_dataset = train_torch_dataset,
        validation_dataset = val_torch_dataset,
        test_dataset = test_torch_dataset,
    )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = featurize_fn,
        config = model_config,
        scaler = None,
    )
    
    MLPMethod.k_fold_training(model)
    MLPMethod.save_model(model)
    accuracy = MLPMethod.check_model_accuracy(model)
