import tyro
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import yaml
import warnings
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict

import lele, bio
from lele.Path import P
from lele.String import get_substring 
from bio.Bioinformatics import Smile
from bio.__global__ import RESULTS_DIR
from loguru import logger
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

SAVE_DIR = RESULTS_DIR / "find_polymer_for_target_molecule"

@dataclass
class PSmileGeneratorConfig():
    model_pt_file: Path = RESULTS_DIR / "pee_smiles_generator" / "pretrained_model_best.pt"
    polymers_to_generate_per_loop: int = 1024
    batch_size: int = 256
    temperature: float = 1.0 # 0.8 = conservative, 1.0 = standard, 1.2 = creative/chaotic
    max_new_tokens: int = 128

@dataclass
class FilterConfig(bio.pee_smiles_filter.FilterConfig):
    """Local subclass to hide internal filtering options from the Tyro CLI."""
    csv_file: tyro.conf.Suppress[Optional[Path]] = None
    save_dir: tyro.conf.Suppress[Optional[Path]] = None
    target_molecule_name: tyro.conf.Suppress[Optional[str]] = None
    target_molecule: tyro.conf.Suppress[Optional[str]] = None
    water_ph: tyro.conf.Suppress[Optional[float]] = None
    csv_train_data: tyro.conf.Suppress[Optional[Path]] = None
    max_size: tyro.conf.Suppress[Optional[int]] = None
    column_name: tyro.conf.Suppress[Optional[str]] = None

@dataclass
class Config:
    target_molecule: tyro.conf.Positional[str]
    target_capacity: float = 1.0
    concentration: float = 12.5
    water_ph: float = 8.2
    seed: int = 42
    save_dir: Path = SAVE_DIR
    model_config: PSmileGeneratorConfig = field(default_factory=PSmileGeneratorConfig)
    filter_config: FilterConfig = field(default_factory=FilterConfig)

def main():
    bio.setup_loguru()
    config = tyro.cli(Config)
    run_with_config(config)

def test_():
    config = Config(target_molecule="metformin")
    config.model_config.polymers_to_generate_per_loop = 16
    config.model_config.batch_size = 4
    run_with_config(config)
    

def run_with_config(config: Config):
    polymer = find_polymer_for_target_molecule(config)
    bio.predict_absorbant_polymer_for_filtered_candidates.plot_capacity_vs_concentration(
        drug_name = config.target_molecule,
        polymer = polymer, 
        drug = bio.get_smiles_from_name(config.target_molecule), 
        concentration = np.linspace(0, 50, 100).tolist(), 
        water_ph = config.water_ph,
        save_dir = get_actual_save_dir(config),
    )
    bio.predict_absorbant_polymer_for_filtered_candidates.plot_capacity_vs_ph(
        drug_name = config.target_molecule,
        polymer = polymer, 
        drug = bio.get_smiles_from_name(config.target_molecule), 
        water_ph = np.linspace(0, 14, 100).tolist(), 
        concentration = config.concentration,
        save_dir = get_actual_save_dir(config),
    )
    

def get_actual_save_dir(config):
    return config.save_dir / f"{config.target_molecule}"

def append_to_csv(df, csv_path):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    df.to_csv(csv_path, mode='a', index=False, na_rep='NaN', header=write_header)

def find_polymer_for_target_molecule(
    config: Config, 
    is_polymer_valid: Callable = bio.Bioinformatics.is_psmiles_string_valid
):
    bio.ML.set_seed(config.seed)
    
    save_dir = get_actual_save_dir(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cleaning up old run files in {save_dir}...")
    for file in save_dir.glob("*.csv"):
        try:
            file.unlink()
            logger.debug(f"Deleted old file: {file.name}")
        except Exception as e:
            logger.warning(f"Could not delete {file.name}: {e}")
    
    config_yaml_path = save_dir / "config.yaml"
    yaml.SafeDumper.add_multi_representer(
        Path, 
        lambda dumper, data: dumper.represent_str(str(data))
    )
    formatted_config = yaml.safe_dump(
        asdict(config), 
        default_flow_style=False, 
        sort_keys=False
    )
    with open(config_yaml_path, "w") as f:
        f.write(formatted_config)
    logger.info(f"Saved run configuration to {config_yaml_path.name}")
        
    column_name = "valid_psmiles"
    config.filter_config.target_molecule_name = config.target_molecule
    config.filter_config.water_ph = config.water_ph
    model, dataset = load_model_and_dataset(config.model_config)
    pscp = bio.predict_absorbant_polymer_for_filtered_candidates.get_trained_model()
    seen_smiles = set()
    while True:
        config.filter_config.column_name = column_name # needs to be set before each run
        generated_polymers = generate_polymers(model, dataset, config.model_config)
        polymer_count = len(generated_polymers)
        logger.trace(f"generated_polymers:\n{generated_polymers}")
        unique_new_polymers = [p for p in generated_polymers if p not in seen_smiles]
        seen_smiles.update(unique_new_polymers)
        df = pd.DataFrame({"PSMILES": unique_new_polymers})
        logger.info(f"Dropped: {polymer_count - len(unique_new_polymers)} non-unique polymers")
        polymer_count = len(unique_new_polymers)
        if df.empty: continue
        append_to_csv(df, save_dir / "01_generated_polymers.csv")
        
        df[column_name] = df["PSMILES"].apply(is_polymer_valid)
        df = df[df[column_name] == True].copy()
        df[column_name] = df["PSMILES"]
        logger.trace(f"valid_polymers:\n{df}")
        logger.info(f"Dropped: {polymer_count - len(df)} invalid polymers")
        polymer_count = len(df)
        if df.empty: continue
        append_to_csv(df, save_dir / "02_valid_polymers.csv")
        
        df = bio.pee_smiles_filter.run_for_dataframe(df, config.filter_config)
        df = bio.pee_smiles_filter.clean_output_df(df)
        logger.trace(f"filtered_polymers:\n{df}")
        # logger.info(f"Dropped: {polymer_count - len(df)} / {polymer_count} invalid polymers")
        # polymer_count = len(df)
        if df.empty: continue
        append_to_csv(df, save_dir / "03_filtered_polymers.csv")
        
        df = bio.predict_absorbant_polymer_for_filtered_candidates.from_df(
            trained_model = pscp,
            df = df,
            water_ph = config.water_ph,
            concentration = config.concentration,
        )
        logger.trace(f"generated_polymers with prediction:\n{df}")
        if df.empty: continue
        append_to_csv(df, save_dir / "04_predicted_capacities.csv")
        
        df = df.sort_values(by="PREDICTED_CAPACITY", ascending=False).reset_index(drop=True)
        best_candidate = df.iloc[0]
        best_capacity = best_candidate["PREDICTED_CAPACITY"]
        best_polymer_smiles = best_candidate["POLYMER_USED"]
        logger.info(f"Best predicted capacity: {best_capacity:.4f} (Target: {config.target_capacity})")
        if best_capacity >= config.target_capacity:
            logger.success(f"Target reached! Found polymer: {best_polymer_smiles} (for target molecule '{config.target_molecule}') with capacity: {best_capacity:.4f}")
            return best_polymer_smiles


def load_model_and_dataset(model_config: PSmileGeneratorConfig) -> bio.MinGPT:
    warnings.filterwarnings("ignore", ".*'pin_memory' argument is set as true.*") # cannot change pin_memory settings.
    logger.debug(f"model_config: {model_config}")
    assert model_config.model_pt_file.exists(), f"model_pt_file not found: {model_config.model_pt_file}"
    dir = model_config.model_pt_file.parent
    config_path = dir / "model_config_used.jsonc"
    assert config_path.exists(), f"model_config_used.jsonc not found: {config_path}"
    
    logger.info(f"Loading config from {config_path.name}...")
    train_config = bio.MinGPT.ModelConfig.load(config_path, add_unique_id=False)
    device = bio.ML.get_torch_device()
    
    logger.info("Reloading tokenizer...")
    unlabeled_smiles = bio.Dataset.UnlabeledSmiles.from_config(train_config.dataset)
    dataset = unlabeled_smiles.to_torch_dataset(block_size=train_config.block_size)

    logger.info("Loading model weights...")
    model = bio.MinGPT.get_model_from_config_and_dataset(train_config, dataset)
    state_dict = torch.load(model_config.model_pt_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model, dataset


def generate_polymers(
    model: bio.MinGPT,
    dataset: bio.Dataset.UnlabeledSmiles,
    model_config: PSmileGeneratorConfig,
):
    device = bio.ML.get_torch_device()
    start_token_id = dataset.tokenizer.cls_token_id
    logger.info(f"Starting generation of {model_config.polymers_to_generate_per_loop} polymers...")
    generated_polymers = []
    with torch.no_grad():
        generated_count = 0
        num_smile_valid = 0
        while generated_count < model_config.polymers_to_generate_per_loop:
            current_batch_size = min(model_config.batch_size, model_config.polymers_to_generate_per_loop - generated_count)
            x = torch.full((current_batch_size, 1), start_token_id, dtype=torch.long).to(device)
            y = model.generate(
                x, 
                max_new_tokens=model_config.max_new_tokens, # Or custom length
                temperature=model_config.temperature, 
                do_sample=True, 
                top_k=None # You can add top_k=40 to reduce weird molecules
            )
            for row in y:
                indices = row.tolist()
                decoded_str = dataset.tokenizer.decode(indices)
                # Sanitize: Extract content between [CLS] and [SEP]
                # Note: MinGPT generate includes the input, so it starts with [CLS]
                psmiles_str = get_substring(decoded_str, "[CLS]", "[SEP]")
                if not psmiles_str: continue
                psmiles_str = psmiles_str.replace(" ", "")
                psmile = bio.Bioinformatics.Smile(psmiles_str)
                if not psmile: continue
                logger.bind(type="GENERATED_POLYMER").trace(psmile)
                generated_polymers.append(psmile)
                generated_count += 1
            logger.debug(f"Generated {generated_count}/{model_config.polymers_to_generate_per_loop}")
    logger.success(f"Done! Generated {model_config.polymers_to_generate_per_loop} polymers")
    return generated_polymers
