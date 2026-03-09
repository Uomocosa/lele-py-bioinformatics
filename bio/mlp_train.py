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
from loguru import logger

MLP_DIR = BIOINFORMATICS_DIR / "MLP_checkpoints"
MLP_TEST_DIR = BIOINFORMATICS_DIR / "MLP_checkpoints_test"
TRAIN_DATASET = DATASETS_DIR / 'PDCC' / 'converted_pdcc_train.csv'
VAL_DATASET = DATASETS_DIR / 'PDCC' / 'converted_pdcc_validation.csv'
TEST_DATASET = DATASETS_DIR / 'PDCC' / 'converted_pdcc_test.csv'

@dataclass
class CommonDatasetConfig(PDCC.Config):
    csv_file: Optional[Path] = None
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_size: Optional[int] = None
    seed: int = 42
    
@dataclass
class ModelConfig(MLP.Config):
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    # hidden_dims: list = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    criterion: nn.Module = nn.MSELoss()
    epochs: int = 10000
    batch_size: int = 16
    learning_rate: float = 1e-3
    early_stop_patience: int = 1000
    best_model_save_dir: Optional[Path] = MLP_DIR
    seed: int = 42


DEFAULT_CAPPING_ATOMS = {"H": 1, "O": 8}
@dataclass
class FeaturizeOptions(PDCCMethod.featurize_v2.Options):
    train_data: Optional[pd.DataFrame] = None
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS) 
    ph_min: float = 7.0
    ph_max: float = 7.4
    precision: float = 0.5
    molecule_multi_featurizer = PDCCMethod.featurize_v2.MOLECULE_MULTI_FEATURIZER
    polymer_multi_featurizer = PDCCMethod.featurize_v2.POLYMER_MULTI_FEATURIZER
    sidechain_multi_featurizer = PDCCMethod.featurize_v2.SIDECHAIN_MULTI_FEATURIZER
    backbone_multi_featurizer = PDCCMethod.featurize_v2.BACKBONE_MULTI_FEATURIZER
    
def main(): 
    # pixi run mlp_train
    lele.Loguru.simple_format()
    dataset_config = tyro.cli(DatasetConfig)
    model_config = tyro.cli(ModelConfig)
    featurize_options = tyro.cli(FeaturizeOptions)
    featurize = lambda df: PDCCMethod.featurize_v2(df, featurize_options)
    run_with_config(
        dataset_config,
        model_config,
        featurize,
    )


def test_():
    lele.Loguru.simple_format()
    dataset_config = CommonDatasetConfig()
    model_config = ModelConfig()
    featurize_options = FeaturizeOptions()
    featurize = lambda df: PDCCMethod.featurize_v2(df, featurize_options)
    run_with_config(
        dataset_config,
        model_config,
        featurize,
        scaler = StandardScaler(),
    )


    
def run_with_config(
    dataset_config: PDCC.Config, 
    model_config: MLP.Config,
    featurize_fn: Optional[Callable[pd.DataFrame, pd.DataFrame]],
    scaler = StandardScaler(),
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

    if scaler is not None:
        scaler.fit(train_torch_dataset.X)
        train_torch_dataset.transform(scaler)
        val_torch_dataset.transform(scaler)
        test_torch_dataset.transform(scaler)
    
    splitted_dataset = bio.Dataset.Splitted.from_datasets(
        train_dataset = train_torch_dataset,
        validation_dataset = val_torch_dataset,
        test_dataset = test_torch_dataset,
    )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = featurize_fn,
        scaler = scaler,
        config = model_config,
    )
    MLPMethod.train_model(model)
    MLPMethod.save_model(model)
    accuracy = MLPMethod.check_model_accuracy(model)
    # logger.info(f"model accuracy: {accuracy}")
    logger.info(f"Train dataset size: {len(model.data.train)}")
    logger.info(f"Validation dataset size: {len(model.data.validation)}")
    logger.info(f"Test dataset size: {len(model.data.test)}")
    
    input_df = pd.DataFrame({
        'POLYMER_USED': ["*/CCC[Fe]CCCC(=O)OCCCCOCCCNCC(*)=O"],
        'DRUG': ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        'PH': [7.0],
        'CONCENTRATION': ["12.5"],
    })
    
    prediction = model.predict(input_df)
    logger.info(f"Predicted Capacity: {prediction:.4f}")
