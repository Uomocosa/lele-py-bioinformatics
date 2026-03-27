from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tyro
import yaml
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import bio
from bio.ML import MLP
from bio.Bioinformatics import PeeSmileCapacityPredictor
from bio.__global__ import RESULTS_DIR  # Make sure this is imported!
from loguru import logger

CANDIDATES_DIR = RESULTS_DIR / "filtered_synthetic_candidates"
PREDICTIONS_DIR = RESULTS_DIR / "predicted_capacities"

@dataclass
class CLIArgs:
    target_molecule: str
    base_concentration: float = 12.5

def main():
    bio.setup_loguru()
    cli_args = tyro.cli(CLIArgs)
    predict_single_candidate(cli_args.target_molecule, cli_args.base_concentration)

import pytest
@pytest.mark.above10s
def test_predict_aspirin():
    # pixi run pytest -rFP -q -s bio\predict_absorbant_polymer_for_filtered_candidates.py::test_predict_aspirin -o "addopts="
    bio.setup_loguru()
    predict_single_candidate("aspirin")
    
import pytest
@pytest.mark.above10s
def test_predict_lisinopril():
    # pixi run pytest -rFP -q -s bio\predict_absorbant_polymer_for_filtered_candidates.py::test_predict_aspirin -o "addopts="
    bio.setup_loguru()
    predict_single_candidate("lisinopril")
    
import pytest
@pytest.mark.above10s
def test_predict_metformin():
    # pixi run pytest -rFP -q -s bio\predict_absorbant_polymer_for_filtered_candidates.py::test_predict_metformin -o "addopts="
    bio.setup_loguru()
    predict_single_candidate("metformin")

    
def get_trained_model() -> MLP:
    pscp = PeeSmileCapacityPredictor()
    trained_model = pscp.load_trained_model()
    assert trained_model is not None, "Failed to load trained model"
    trained_model.eval()
    return trained_model

    
def predict_single_candidate(
    target_molecule_name: str, 
    base_concentration: float = 12.5,
    trained_model: Optional[MLP] = None, 
):
    assert CANDIDATES_DIR.exists()
    csv_file = CANDIDATES_DIR / f"target_{target_molecule_name}.csv"
    assert csv_file.exists(), f"No candidate CSV file found in {CANDIDATES_DIR}.\nYou need to run first 'pixi run pee_smile_filter --target_molecule {target_molecule_name}'"
    
    yaml_file = CANDIDATES_DIR / f"target_{target_molecule_name}_filter_config.yaml"
    if yaml_file.exists():
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
            # Default to 8.2 if water_ph isn't found in the yaml for some reason
            water_ph = config_dict.get("water_ph", 8.2) 
    else:
        logger.warning(f"No YAML config found at {yaml_file}. Defaulting water_ph to 8.2")
        water_ph = 8.2
    
    if not trained_model: trained_model = get_trained_model()
    logger.info(f"Predicting for target molecule: {target_molecule_name}")
    df = predict_absorbant_polymer_for_filtered_candidates(
        trained_model = trained_model,
        csv_file = csv_file,
        water_ph = water_ph,
        concentration = base_concentration
    )
    if df.empty: return
    logger.info(f"Best predicted capacity for {target_molecule_name}: {df['PREDICTED_CAPACITY'].max()} (from {df.shape[0]} candidates)")
    PREDICTIONS_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(PREDICTIONS_DIR / f"best_predictions_for_{target_molecule_name}.csv", index=False)
    best_prediction = df.loc[df['PREDICTED_CAPACITY'].idxmax()]
    polymer = best_prediction['POLYMER_USED']
    drug = best_prediction['DRUG']
    plot_capacity_vs_concentration(
        drug_name = target_molecule_name,
        polymer = polymer, 
        drug = drug, 
        concentration = np.linspace(0, 50, 100).tolist(), 
        water_ph = water_ph,
    )
    plot_capacity_vs_ph(
        drug_name = target_molecule_name,
        polymer = polymer, 
        drug = drug, 
        water_ph = np.linspace(0, 14, 100).tolist(), 
        concentration = base_concentration,
    )
        

def predict_absorbant_polymer_for_filtered_candidates(
    trained_model: MLP,
    csv_file: Path,
    water_ph: float = 8.2, 
    concentration: float = 12.5
) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if df.empty:
        logger.warning(f"The filtered candidates dataframe for {csv_file.stem} is empty.")
        return pd.DataFrame()
    return from_df(trained_model, df, water_ph, concentration)
        
def from_df(
    trained_model: MLP,
    df: pd.DataFrame,
    water_ph: float = 8.2, 
    concentration: float = 12.5
) -> pd.DataFrame:
    input_df = df[['POLYMER_USED', 'DRUG']].copy()
    input_df['WATER_PH'] = water_ph
    input_df['CONCENTRATION'] = concentration 
    logger.info(f"Predicting capacities for {len(input_df)} candidates in batch...")
    predictions = trained_model.predict(input_df)
    out_df = input_df.copy()
    out_df['PREDICTED_CAPACITY'] = predictions
    out_df = out_df.sort_values(by='PREDICTED_CAPACITY', ascending=False).reset_index(drop=True)
    return out_df



def plot_capacity_vs_concentration(
    drug_name: str,
    polymer: str, 
    drug: str, 
    concentration: list[float],
    water_ph: float = 8.2,
    save_dir: Path = PREDICTIONS_DIR,
) -> None:
    """Plots predicted capacity for a single polymer/drug pair across varying concentrations."""
    logger.info(f"Plotting capacity vs. concentration for {drug_name} (Fixed pH: {water_ph})")
    
    # 1. Load Model (Loads once per function call)
    pscp = PeeSmileCapacityPredictor()
    model = pscp.load_trained_model()
    if hasattr(model, 'eval'): model.eval()
    
    # 2. Prepare Batch Data
    df = pd.DataFrame({
        'POLYMER_USED': [polymer] * len(concentration),
        'DRUG': [drug] * len(concentration),
        'WATER_PH': [water_ph] * len(concentration),
        'CONCENTRATION': concentration
    })
    
    # 3. Predict
    df['PREDICTED_CAPACITY'] = model.predict(df)
    
    # 4. Plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='CONCENTRATION', y='PREDICTED_CAPACITY', marker='o', color='b', linewidth=2)
    plt.suptitle('Capacity vs. Concentration', fontsize=14, y=0.98)
    plt.title(f'Drug: {drug_name} | Fixed pH: {water_ph}\nPolymer PSMILES: {polymer}', fontsize=10, color='dimgray')
    plt.xlabel('Concentration')
    plt.ylabel('Predicted Capacity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 5. Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"plot_conc_{drug_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved concentration plot to {save_path}")



def plot_capacity_vs_ph(
    drug_name: str,
    polymer: str, 
    drug: str, 
    water_ph: list[float],
    concentration: float = 12.5,
    save_dir: Path = PREDICTIONS_DIR,
) -> None:
    """Plots predicted capacity for a single polymer/drug pair across varying pH levels."""
    logger.info(f"Plotting capacity vs. pH for {drug_name} (Fixed Conc: {concentration})")
    
    # 1. Load Model
    pscp = PeeSmileCapacityPredictor()
    model = pscp.load_trained_model()
    if hasattr(model, 'eval'): model.eval()
    
    # 2. Prepare Batch Data
    df = pd.DataFrame({
        'POLYMER_USED': [polymer] * len(water_ph),
        'DRUG': [drug] * len(water_ph),
        'WATER_PH': water_ph,
        'CONCENTRATION': [concentration] * len(water_ph)
    })
    
    # 3. Predict (This will trigger the featurizer to recalculate protonation states per pH)
    df['PREDICTED_CAPACITY'] = model.predict(df)
    
    # 4. Plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='WATER_PH', y='PREDICTED_CAPACITY', marker='o', color='r', linewidth=2)
    plt.suptitle('Capacity vs. Water pH', fontsize=14, y=0.98)
    plt.title(f'Drug: {drug_name} | Fixed Conc: {concentration}\nPolymer PSMILES: {polymer}', fontsize=10, color='dimgray')
    plt.xlabel('Water pH')
    plt.ylabel('Predicted Capacity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 5. Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"plot_ph_{drug_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved pH plot to {save_path}")
