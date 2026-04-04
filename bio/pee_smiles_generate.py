import tyro
import torch
import time, warnings
from pathlib import Path
from loguru import logger
from typing import Optional, Callable
from dataclasses import dataclass
import lele, bio
from bio.ML.__global__ import HELPER_DIR
from bio.__global__ import RESULTS_DIR
from lele.Path import P
from lele.String import get_substring 
from bio.Bioinformatics import Smile
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

SAVE_DIR = RESULTS_DIR / "pee_smiles_generator"
CHECKPOINT_TEST_FOLDER = HELPER_DIR / 'PSMILES_checkpoints_test'

@dataclass
class PSmileGenerateConfig(bio.cacca_generate.GenerateConfig):
    model_dir: Path = SAVE_DIR
    is_smile_valid: Callable[[str], bool] = lambda psmile: bio.Bioinformatics.is_psmiles_string_valid(psmile)

import pytest
@pytest.mark.above10s
def test_():
    # pixi run pytest -rFP -q -s bio\pee_smiles_generate.py -o "addopts="
    config = PSmileGenerateConfig()
    config.smiles_to_generate = 10
    config.batch_size = 1
    config.model_dir = CHECKPOINT_TEST_FOLDER
    bio.cacca_generate.run_with_config(config)
    
def main():
    config = tyro.cli(PSmileGenerateConfig)
    bio.cacca_generate.run_with_config(config)
