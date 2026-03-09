import yaml, json
import tyro
from typing import Optional
from dataclasses import asdict
from itertools import product
from pathlib import Path
import lele, bio
from experiment.__call__ import Experiment, DatasetConfig, ModelConfig, FeaturizerConfig, run
from loguru import logger

FORCE_RERUN = False
EXPERIMENTS = [
    [Experiment(
        name="experiment_baseline",
    ), FORCE_RERUN],
    [Experiment(
        name="experiment_small_data",
        dataset=DatasetConfig(max_size=10),
        model=ModelConfig(epochs=100),
    ), FORCE_RERUN],
    [Experiment(
        name="experiment_mae_loss",
        model=ModelConfig(criterion="mae")
    ), FORCE_RERUN],
]

grid_lrs = [
    1e-3, 5e-4
]
grid_dropouts = [
    0.1, 0.3
]
grid_hidden_dims = [
    [128, 64, 32], 
    [256, 128, 64]
]

GRID_EXPERIMENTS = [
    [Experiment(
        # Dynamically name the experiment based on parameters
        name=f"exp_lr{lr}_drop{drop}_hd{len(hd)}",
        model=ModelConfig(learning_rate=lr, dropout=drop, hidden_dims=hd)
    ), FORCE_RERUN]
    for lr, drop, hd in product(grid_lrs, grid_dropouts, grid_hidden_dims)
]

# Combine all experiments to run
ALL_EXPERIMENTS = EXPERIMENTS + GRID_EXPERIMENTS

def test_(): 
    run_all_experiments()

def run_all_experiments():
    save_dir = lele.P(__file__).parent  
    
    for exp, force_rerun in ALL_EXPERIMENTS:
        logger.info(f"Evaluating {exp.name}...")
        
        if not force_rerun and is_experiment_unchanged(exp, save_dir):
            logger.info(f"⏭️ Skipping {exp.name} - Configuration is unchanged.")
            continue
            
        logger.info(f"🚀 Starting {exp.name}...")
        try:
            run(exp)
            logger.info(f"✅ Successfully finished {exp.name}.")
        except Exception as e:
            logger.error(f"❌ Failed {exp.name} with error: {e}")


def _normalize_config(config_dict: dict) -> dict:
    """Converts a dict with complex objects (like Path) into native strings/types."""
    # A quick trick: dumping to JSON with default=str converts Paths to strings,
    # then loading it back gives us a clean dictionary comparable to parsed YAML.
    return json.loads(json.dumps(config_dict, default=str))

def is_experiment_unchanged(experiment: Experiment, save_dir: Path) -> bool:
    """Checks if the experiment has already been run with the exact same config."""
    config_path = save_dir / experiment.name / "config.yaml"
    
    if not config_path.exists():
        return False
        
    try:
        with open(config_path, "r") as f:
            saved_config = yaml.safe_load(f)
            
        current_config = _normalize_config(asdict(experiment))
        
        return saved_config == current_config
    except Exception as e:
        logger.warning(f"Could not read/compare config for {experiment.name} ({e}). Defaulting to rerun.")
        return False
