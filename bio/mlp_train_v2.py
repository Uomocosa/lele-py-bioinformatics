import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from bio.__global__ import BIOINFORMATICS_DIR, DATASETS_DIR, CONVERTED_PDCC_CSV
from loguru import logger

ML_HELPER_DIR = bio.ML.__global__.HELPER_DIR
MLP_DIR = BIOINFORMATICS_DIR / "MLP_checkpoints"
MLP_TEST_DIR = BIOINFORMATICS_DIR / "MLP_checkpoints_test"

FeaturizerOptions = bio.Dataset.PDCCMethod.featurize.Options

@dataclass
class CommonDatasetConfig(PDCC.Config):
    csv_file: Optional[Path] = CONVERTED_PDCC_CSV
    train_validation_test_pecentages: Tuple[float, float, float] = (1, 0, 0)
    # max_size: Optional[int] = 10
    seed: int = 42
    
@dataclass
class ModelConfig(MLP.Config):
    """
    This settings are taken from the reults of k-fold cross validations
    """
    hidden_dims: list = field(default_factory=lambda: [8, 8, 8, 4])
    dropout: float = 0.2
    criterion: nn.Module = nn.MSELoss()
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-3
    early_stop_patience: int = 300
    best_model_save_dir: Optional[Path] = MLP_DIR
    seed: int = 42

def main(): 
    # pixi run mlp_train_v2
    lele.Loguru.simple_format()
    dataset_config = tyro.cli(DatasetConfig)
    model_config = tyro.cli(ModelConfig)
    featurize_options = tyro.cli(FeaturizerOptions)
    featurize = lambda df: PDCCMethod.featurize_v2(df, featurize_options)
    run_with_config(
        dataset_config,
        model_config,
        featurize,
    )


import pytest
@pytest.mark.above10s
def test_():
    setup_loguru()
    dataset_config = CommonDatasetConfig()
    model_config = ModelConfig()
    featurize_options = FeaturizerOptions()
    featurize = lambda df: PDCCMethod.featurize(df, featurize_options)
    model = run_with_config(
        dataset_config,
        model_config,
        featurize,
        scaler = None,
    )
    plot_aspirin(model)
    plot_metformin(model)
    
    
def plot_aspirin(model):
    drug_name = "aspirin"
    aspirin_smile = "CC(=O)OC1=CC=CC=C1C(=O)O"
    polymers_to_test = {
        "Polymer_A": "*CC(C#N)=C(C=C(C)C(=O)NC(C)N*)CCCCCCC",
        "Polymer_B": "*CC(CO)COC(=O)CCCCCCCCCCC(=O)OCCCN(*)C",
        "Polymer_C": "*CCCCCCCOC(=O)NC(=O)NCCCCCCCNC(=O)OC1CCCN(*)CCOCC1",
        "Polymer_D": "*CCCCCNC(=O)/C=C/C(=N)NCCCCCCCNC(=O)OC=C/C=C(*)CC",
        "Polymer_E": "*CCN(CC#N)C(=O)CCCCOC(=O)/C=CC/C=C/c1c(=O)NC(CCC)c(N(*)C)cc1",
        # Add as many as you want here
    }

    for poly_name, poly_smile in polymers_to_test.items():
        logger.info(f"Generating plots for {poly_name}...")
        plot_capacity_vs_concentration(
            model=model,
            polymer_smile=poly_smile,
            drug_smile=aspirin_smile,
            fixed_water_ph=8.2,
            min_conc=1.0,
            max_conc=50.0,
            num_steps=50,
            save_location=ML_HELPER_DIR / f"{poly_name}_{drug_name}_conc_plot.png"
        )
        plot_capacity_vs_ph(
            model=model,
            polymer_smile=poly_smile,
            drug_smile=aspirin_smile,
            fixed_concentration=12.5,
            min_ph=1.0,
            max_ph=14.0,
            num_steps=50,
            save_location=ML_HELPER_DIR / f"{poly_name}_{drug_name}_ph_plot.png"
        )

def plot_metformin(model):
    drug_name = "metformin"
    metformin_smile = "CC(=O)OC1=CC=CC=C1C(=O)O"
    polymers_to_test = {
        "Polymer_A": "*CC(=O)NCCCCCCCCCCCNC(=O)Nc1ccc(CCc2ccc(NC(=O)CCCCCC(=O)N*)cc2)cc1",
        "Polymer_B": "*CCCCCCCCCC(=O)NNC(=O)CCCSCCCCCC(=O)N*",
        "Polymer_C": "*C(=O)NCCCCNC(=O)c1ccc(C(=O)Nc2ccc(NC(=O)CCCCCCCC(=O)Nc3ccc(O*)cc3)cc2)cc1",
        "Polymer_D": "*CCCCCCCCCCCCCCCOC(=O)CCCCCCC(=O)OCc1ccc(/C=C/c2ccc(C(=O)O*)cc2)cc1",
        "Polymer_E": "*CC(OC)C(COCOC(=O)c1ccc(/C=N/c2ccc(C(=O)O*)cc2)cc1)CCC",
        # Add as many as you want here
    }

    for poly_name, poly_smile in polymers_to_test.items():
        logger.info(f"Generating plots for {poly_name}...")
        plot_capacity_vs_concentration(
            model=model,
            polymer_smile=poly_smile,
            drug_smile=metformin_smile,
            fixed_water_ph=8.2,
            min_conc=1.0,
            max_conc=50.0,
            num_steps=50,
            save_location=ML_HELPER_DIR / f"{poly_name}_{drug_name}_conc_plot.png"
        )
        plot_capacity_vs_ph(
            model=model,
            polymer_smile=poly_smile,
            drug_smile=metformin_smile,
            fixed_concentration=12.5,
            min_ph=1.0,
            max_ph=14.0,
            num_steps=50,
            save_location=ML_HELPER_DIR / f"{poly_name}_{drug_name}_ph_plot.png"
        )
      
    
def plot_capacity_vs_concentration(
    model, 
    polymer_smile: str, 
    drug_smile: str, 
    fixed_water_ph: float, 
    min_conc: float = 0.0, 
    max_conc: float = 100.0, 
    num_steps: int = 50,
    save_location: Optional[Path] = None,
):
    """Plots predicted capacity against a linspace of concentrations."""
    
    # Generate the linspace for concentration
    concentrations = np.linspace(min_conc, max_conc, num_steps)
    
    # Create the input DataFrame
    input_df = pd.DataFrame({
        'POLYMER_USED': [polymer_smile] * num_steps,
        'DRUG': [drug_smile] * num_steps,
        'WATER_PH': [fixed_water_ph] * num_steps,
        'CONCENTRATION': concentrations,
    })
    
    predictions = [model.predict(input_df.iloc[[i]]) for i in range(len(input_df))]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(concentrations, predictions, color='blue', marker='o', markersize=3, label=f'pH = {fixed_water_ph}')
    plt.suptitle(f'Capacity vs Concentration:', fontsize=14, fontweight='bold', y=0.98)
    subtitle_text = f"Polymer: {polymer_smile}\nDrug: {drug_smile}"
    plt.title(subtitle_text, fontsize=9, color='dimgrey', pad=10)
    plt.xlabel('Concentration')
    plt.ylabel('Predicted Capacity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if save_location:
        save_path = Path(save_location)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Concentration plot to {save_path}")
    # plt.show()
    plt.close()

def plot_capacity_vs_ph(
    model, 
    polymer_smile: str, 
    drug_smile: str, 
    fixed_concentration: float, 
    min_ph: float = 1.0, 
    max_ph: float = 14.0, 
    num_steps: int = 50,
    save_location: Optional[Path] = None,
):
    """Plots predicted capacity against a linspace of water pH values."""
    
    # Generate the linspace for pH
    phs = np.linspace(min_ph, max_ph, num_steps)
    
    # Create the input DataFrame
    input_df = pd.DataFrame({
        'POLYMER_USED': [polymer_smile] * num_steps,
        'DRUG': [drug_smile] * num_steps,
        'WATER_PH': phs,
        'CONCENTRATION': [fixed_concentration] * num_steps,
    })
    
    predictions = [model.predict(input_df.iloc[[i]]) for i in range(len(input_df))]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(phs, predictions, color='red', marker='s', markersize=3, label=f'Conc = {fixed_concentration}')
    plt.suptitle(f'Capacity vs Water pH:', fontsize=14, fontweight='bold', y=0.98)
    subtitle_text = f"Polymer: {polymer_smile}\nDrug: {drug_smile}"
    plt.title(subtitle_text, fontsize=9, color='dimgrey', pad=10)
    plt.xlabel('Water pH')
    plt.ylabel('Predicted Capacity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if save_location:
        save_path = Path(save_location)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Concentration plot to {save_path}")
    # plt.show()
    plt.close()
        
        
def setup_loguru():
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            "bio.ML.MLPMethod.train_model": "WARNING",
        },
        level = "INFO"
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
    
    dataset = PDCC(dataset_config)
    dataset.featurize = featurize_fn
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
    
    if scaler is not None:
        scaler = splitted_dataset.scale(
            feature_col_indexes = range(torch_dataset.num_features),
            scaler_fn = StandardScaler()
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

    logger.info(f"Train dataset size: {len(model.data.train)}")
    logger.info(f"Validation dataset size: {len(model.data.validation)}")
    logger.info(f"Test dataset size: {len(model.data.test)}")

    return model
