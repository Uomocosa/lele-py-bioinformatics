import tyro
import torch
import time, warnings
from pathlib import Path
from loguru import logger
from typing import Optional, Callable
from dataclasses import dataclass
import lele, bio
from lele.Path import P
from lele.String import get_substring 
from bio.Bioinformatics import Smile
from bio.MinGPT.__global__ import CHECKPOINT_FOLDER
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)


@dataclass
class GenerateConfig():
    smiles_to_generate: int = 10000
    model_dir: Path = CHECKPOINT_FOLDER
    batch_size: int = 1
    temperature: float = 1.0 # 0.8 = conservative, 1.0 = standard, 1.2 = creative/chaotic
    max_new_tokens: int = 128
    use_best_model_in_subfolders: bool = True
    is_smile_valid: Callable[[Smile], bool] = lambda smile: smile.is_valid

import pytest
@pytest.mark.above10s
def test_():
    config = GenerateConfig()
    config.smiles_to_generate = 3
    run_with_config(config)
    
def main():
    config = tyro.cli(GenerateConfig)
    run_with_config(config)

def run_with_config(config: GenerateConfig):
    warnings.filterwarnings("ignore", ".*'pin_memory' argument is set as true.*") # cannot change pin_memory settings.
    assert config.model_dir.exists()
    if config.use_best_model_in_subfolders: 
        model_file = find_latest_best_model(config.model_dir)
        config.model_dir = model_file.parent
    else: 
        model_files = config.model_dir.glob("*.pt")
        assert len(model_files) > 0, "No models found in the directory"
        assert len(model_files) == 1, "Found multiple models in the directory"
        model_file = model_files[0]
    output_dir = config.model_dir / get_output_dir_name(config)
    logger.info(f"Generated SMILES will be saved in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    create_jsonc_config(config, output_dir)
    output_file = output_dir / "generated_smiles.csv"
    setup_genereted_smile_logger(output_file)
    lele.Loguru.simple_format()
    lele.Loguru.add_csv_logger(
        csv_file = output_file,
        csv_header = "generated_smiles",
        label = "GENERATED_SMILE",
    )
    lele.Loguru.add_csv_logger(
        csv_file = output_file.parent/"valid_smiles.csv",
        csv_header = "valid_smiles",
        label = "VALID_SMILE",
    )
    logger.debug(f"model_file: {model_file}")

    dir = model_file.parent
    config_path = dir / "model_config_used.jsonc"
    logger.info(f"Loading config from {config_path.name}...")
    train_config = bio.MinGPT.ModelConfig.load(config_path, add_unique_id=False)
    device = bio.ML.get_torch_device()
    
    # IMPORTANT: This should be changed, it is stupid the need to re-initialize the dataset every time
    logger.info("Reloading tokenizer...")
    unlabeled_smiles = bio.Dataset.UnlabeledSmiles.from_config(train_config.dataset)
    dataset = unlabeled_smiles.to_torch_dataset(block_size=train_config.block_size)

    logger.info("Loading model weights...")
    model = bio.MinGPT.get_model_from_config_and_dataset(train_config, dataset)
    state_dict = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    generated_valid_smiles = []
    start_token_id = dataset.tokenizer.cls_token_id
    logger.info(f"Starting generation of {config.smiles_to_generate} molecules...")

    with torch.no_grad():
        generated_count = 0
        num_smile_valid = 0
        while generated_count < config.smiles_to_generate:
            # Create a batch of start tokens
            current_batch_size = min(config.batch_size, config.smiles_to_generate - generated_count)
            
            # Shape: (Batch_Size, 1) -> [[12], [12], [12]...]
            x = torch.full((current_batch_size, 1), start_token_id, dtype=torch.long).to(device)

            # Generate
            t0 = time.perf_counter()
            y = model.generate(
                x, 
                max_new_tokens=config.max_new_tokens, # Or custom length
                temperature=config.temperature, 
                do_sample=True, 
                top_k=None # You can add top_k=40 to reduce weird molecules
            )
            dt = time.perf_counter() - t0

            # Decode Batch
            for row in y:
                indices = row.tolist()
                decoded_str = dataset.tokenizer.decode(indices)
                
                # Sanitize: Extract content between [CLS] and [SEP]
                # Note: MinGPT generate includes the input, so it starts with [CLS]
                smiles_str = get_substring(decoded_str, "[CLS]", "[SEP]")
                if not smiles_str: continue
                smiles_str = smiles_str.replace(" ", "")
                smile = bio.Bioinformatics.Smile(smiles_str)
                # smile = bio.Bioinformatics.Smile(smiles_str).canonicalize() # cannot do this, it returns single atoms like: C, N, ...
                
                if smile: 
                    logger.bind(type="GENERATED_SMILE").trace(smile)
                    generated_count += 1
                if config.is_smile_valid(smile): 
                    logger.bind(type="VALID_SMILE").info(smile)
                    num_smile_valid += 1

            logger.debug(f"Generated {generated_count}/{config.smiles_to_generate} (Batch time: {dt:.2f}s)")
            
    logger.success(f"Done! Generated {config.smiles_to_generate} molecules, {num_smile_valid} valid")

 

def find_latest_best_model(dir: Path) -> Path:
    models = dir.rglob("*pretrained_model_best.pt")
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    # logger.debug(f"latest_model: {latest_model}")
    return latest_model

   
def get_output_dir_name(config: GenerateConfig):
   temp_str = f"{int(config.temperature * 100000000):09d}"
   return f"generate_mnt{config.max_new_tokens}_t{temp_str}"


def create_jsonc_config(config: GenerateConfig, dir: Path):
   config_dict = {
       "temperature": config.temperature,
       "max_new_tokens": config.max_new_tokens,
   }
   config_file = dir / "generate_config.jsonc"
   header_comment  = "Smiles generated with the following options\n"
   header_comment += "// NOTE! If you generate other SMILES with these options and same model, "
   header_comment += "they will be saved in this folder."
   lele.Json.save_dict_to_jsonc_file(
       config_dict, config_file, header=header_comment
   )


def setup_genereted_smile_logger(log_file_path: Path):
    assert log_file_path.suffix == ".csv"
    csv_format = "{message}" 
    logger.add(
        log_file_path, 
        format=csv_format, 
        filter=lambda record: record["extra"].get("type") == "GENERATED_SMILE",
        level="TRACE"
    )
    if not log_file_path.exists():
        with open(log_file_path, "w") as f:
            f.write("generated_smile\n")
