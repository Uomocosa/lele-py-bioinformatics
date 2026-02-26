import lele
import tyro
from mingpt.model import GPT
import warnings 
import time, math
import torch
from dataclasses import dataclass, field
from typing import Optional, Any, Annotated, get_type_hints
from mingpt.trainer import Trainer
import lele, bio
from lele.Json import pretty_json
from lele.String import unique, get_substring
from lele.Metaprogramming import CSV_Logger
from bio.ML import set_seed, get_torch_device
from bio.MinGPT.__global__ import MIN_GPT_CONFIG_FILE
from bio.Dataset.__global__ import PI1M
from pathlib import Path
from loguru import logger
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

CHECKPOINT_FOLDER = lele.P(r"./PSMILES_checkpoints") 
CHECKPOINT_TEST_FOLDER = lele.P(r"./PSMILES_checkpoints_test") 

def main():
    ModelConfig = bio.MinGPT.ModelConfig.ModelConfig
    Options = bio.MinGPT.ModelConfig.Options
    DatasetConfig = bio.Dataset.Config.Config
    
    @dataclass
    class PSmileModelConfig(ModelConfig):
        options: tyro.conf.OmitArgPrefixes[Options] = field(default_factory=Options)
        dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(csv_file=PI1M))
        
    config = tyro.cli(PSmileModelConfig)
    config.options.checkpoint_dir = config.options.checkpoint_dir/lele.String.unique()
    config.options.checkpoint_dir.mkdir(exist_ok=False, parents=True)
    config.dataset
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Device: CPU')
    print(f"Running with epochs: {config.epochs}")
    print(f"Checkpoint dir: {config.options.checkpoint_dir}")
    bio.cacca_train.run_with_config(config)

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
            csv_file=PI1M,
            max_size=1000,
        ),
        options=bio.MinGPT.ModelConfig.Options(
            save_checkpoint_every_n_iters=10
        ),
    )
    simple_config.options.checkpoint_dir = CHECKPOINT_TEST_FOLDER/lele.String.unique()
    simple_config.options.checkpoint_dir.mkdir(exist_ok=False, parents=True)
    bio.cacca_train.run_with_config(simple_config)
