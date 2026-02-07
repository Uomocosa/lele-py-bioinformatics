import lele
import tyro
from mingpt.model import GPT
import warnings 
import time, math
import torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, Annotated, get_type_hints
from mingpt.trainer import Trainer
import lele, bio
from lele.Json import pretty_json
from lele.Path import P
from lele.String import unique, get_substring
from lele.Metaprogramming import CSV_Logger
from bio.ML import set_seed, get_torch_device
from bio.Bioinformatics import Smile
from bio.MinGPT.__global__ import MIN_GPT_CONFIG_FILE
from bio.Dataset.__global__ import ZINC_BASE_CSV
from loguru import logger
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

CHECKPOINT_FOLDER = lele.P(r"./SMILES_checkpoints") 
CHECKPOINT_TEST_FOLDER = lele.P(r"./SMILES_checkpoints_test") 

def main():
    ModelConfig = bio.MinGPT.ModelConfig.ModelConfig
    Options = bio.MinGPT.ModelConfig.Options.Options
    DatasetConfig = bio.Dataset.Config.Config
        
    basic_dataset_config = DatasetConfig(
        csv_file=ZINC_BASE_CSV,
        train_validation_test_pecentages=(0.6, 0.2, 0.2),
        max_size=None,
        process_raw_smiles=canonicalize
    )
    
    @dataclass
    class SmileModelConfig(ModelConfig):
        options: tyro.conf.OmitArgPrefixes[Options] = field(default_factory=Options)
        dataset: DatasetConfig = field(default_factory=lambda: basic_dataset_config)
        
    config = tyro.cli(SmileModelConfig)
    config.options.checkpoint_dir = config.options.checkpoint_dir/lele.String.unique()
    config.options.checkpoint_dir.mkdir(exist_ok=False, parents=True)
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Device: CPU')
    print(f"Running with epochs: {config.epochs}")
    print(f"Checkpoint dir: {config.options.checkpoint_dir}")
    run_with_config(config)

import pytest
@pytest.mark.above10s    
def test_():
    simple_config = bio.MinGPT.ModelConfig(
        model_type='gpt-nano',
        learning_rate=3e-3,
        batch_size=16,
        num_workers=0,
        early_stop_patience=2,
        dataset=bio.Dataset.Config(
            max_size=1000,
            process_raw_smiles=canonicalize,
        ),
        options=bio.MinGPT.ModelConfig.Options(
            save_checkpoint_every_n_iters=10
        ),
    )
    simple_config.options.checkpoint_dir = CHECKPOINT_TEST_FOLDER/lele.String.unique()
    simple_config.options.checkpoint_dir.mkdir(exist_ok=False, parents=True)
    run_with_config(simple_config)


def canonicalize(raw_smiles: pd.Series) -> pd.Series:
    logger.debug("Canonicalizing SMILES...")
    smiles = raw_smiles.apply(lambda s: Smile(s).canonicalize())
    smiles = smiles.dropna().drop_duplicates()
    return smiles
    
    
def run_with_config(config: bio.MinGPT.ModelConfig):
    warnings.filterwarnings("ignore", ".*'pin_memory' argument is set as true.*") # cannot change pin_memory settings.
    lele.Loguru.simple_format()
    config.save_if_requested()
    unlabeled_smiles = bio.Dataset.UnlabeledSmiles.from_config(config.dataset)
    entire_dataset = unlabeled_smiles.to_torch_dataset(block_size=config.block_size)
    model = bio.MinGPT.get_model_from_config_and_dataset(config, entire_dataset)
    
    p_train, p_val, p_test = config.dataset.train_validation_test_pecentages
    dataset = bio.Dataset.split_dataset(entire_dataset, p_train, p_val, p_test, config.seed)
    logger.info(f"Split sizes: Train={len(dataset.train)}, Val={len(dataset.validation)}, Test={len(dataset.test)}")
    
    trainer_config = bio.MinGPT.get_trainer_config_from_model_config_and_dataset(config, dataset.train)
    trainer = Trainer(trainer_config, model, dataset.train)
    trainer.add_callback('on_batch_end', save_model_and_log_loss_data(config))
    trainer.add_callback('on_batch_end', save_best_model(config))
    trainer.add_callback('on_batch_end', early_stopping(config, dataset.validation))
    a = time.perf_counter()
    trainer.run()
    b = time.perf_counter()
    logger.info(f"model trained in {(b-a):.2f} s")
    return
    
    
def save_model_and_log_loss_data(config):
    setup_loss_logger(config.options.checkpoint_dir/"loss_data.csv")
    if config.options.save_checkpoint_every_n_iters: 
        logger.info(f"Checpoints will be saved in:\n\t'{config.options.checkpoint_dir}/'")
    def callback(trainer):
        itr = trainer.iter_num + 1
        n = config.options.save_checkpoint_every_n_iters
        if n and itr % n == 0:
            current_check_point_path = config.options.checkpoint_dir / f"pretrained_model_iter_{itr}.pt"
            current_check_point_path.parent.mkdir(exist_ok=True) 
            torch.save(trainer.model.state_dict(), current_check_point_path)
            logger.debug(f"Saved Checkpoint at iter {itr}")
        if config.options.log_loss_data and trainer.loss:
            csv_line = f"{itr},{trainer.loss.item():.6f}"
            logger.bind(type="LOSS_DATA").trace(csv_line)
    return callback



def setup_loss_logger(log_file_path: Path):
    assert log_file_path.suffix == ".csv"
    csv_format = "{message}" 
    logger.add(
        log_file_path, 
        format=csv_format, 
        filter=lambda record: record["extra"].get("type") == "LOSS_DATA",
        level="TRACE"
    )
    with open(log_file_path, "w") as f:
        f.write("iteration,loss\n")



def save_best_model(config):
    lowest_loss = float('inf')
    iters_since_best_loss = 0
    if config.options.save_best_model: 
        logger.info(f"The best model will be saved as:\n\t'{config.options.checkpoint_dir/'pretrained_model_best.pt'}'")
    def callback(trainer):
        if not config.options.save_best_model: return
        nonlocal lowest_loss, iters_since_best_loss
        current_loss = trainer.loss.item()
        current_iter = trainer.iter_num + 1 # iter_num is 0-indexed
        if current_loss is not None and current_loss < lowest_loss:
            lowest_loss = current_loss
            iters_since_best_loss = 0
            best_model_path = config.options.checkpoint_dir / "pretrained_model_best.pt"
            torch.save(trainer.model.state_dict(), best_model_path)
            logger.debug(f"New best model saved at iter {current_iter} with loss {lowest_loss:.4f}")
        else:
            iters_since_best_loss += 1
    return callback




def early_stopping(
    config: bio.MinGPT.ModelConfig, 
    validation_dataset: torch.utils.data.Dataset,
    check_validation_every_N = 10,
):
    patience = config.early_stop_patience
    if not patience: 
        return lambda t: None # Do nothing if patience is not set
    
    N = check_validation_every_N
    def check_validation(iter: int):
        if iter % N == 0: return True
        else: return False

    # Create a DataLoader for validation (created once to be efficient)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0 # Avoid overhead for small validation checks
    )
    
    best_val_loss = float('inf')
    patience_counter = 0

    # 2. Helper to run the validation loop
    def compute_val_loss(model, device):
        model.eval() # Switch to eval mode (disable dropout, etc.)
        losses = []
        max_batches = 50 # Optimization: Don't validate on *entire* dataset if it's huge
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= max_batches: break
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y) # Forward pass only
                losses.append(loss.item())
                
        model.train() # IMPORTANT: Switch back to training mode
        return sum(losses) / len(losses) if losses else float('inf')

    # 3. The Callback
    def callback(trainer):
        nonlocal best_val_loss, patience_counter
        if not check_validation(trainer.iter_num + 1): return
        val_loss = compute_val_loss(trainer.model, trainer.device)
        logger.info(f"Checking Validation [Iter {trainer.iter_num + 1}] - Validation Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Optional: Save the "best validation" model here if you want
        else:
            patience_counter += N
            if patience_counter >= patience:
                logger.warning(f"EARLY STOPPING: Validation loss hasn't improved for {patience} checks (Current: {val_loss:.4f}, Best: {best_val_loss:.4f})")
                # logger.info(f"EARLY STOPPING: Validation loss hasn't improved for {patience} checks (Current: {val_loss:.4f}, Best: {best_val_loss:.4f})")
                trainer.iter_num = trainer.config.max_iters + 1 # Force the trainer loop to exit
    return callback
