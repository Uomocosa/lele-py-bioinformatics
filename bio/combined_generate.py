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
from bio.ML.__global__ import HELPER_DIR
from bio.__global__ import RESULTS_DIR
from bio.Bioinformatics import Smile
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

SAVE_DIR = RESULTS_DIR / "smiles_and_psmiles_generator"
CHECKPOINT_TEST_FOLDER = HELPER_DIR / 'COMBINED_checkpoints_test'

@dataclass
class CombinedGenerateConfig(bio.cacca_generate.GenerateConfig):
    model_dir: Path = SAVE_DIR
    is_smile_valid: Callable[[str], bool] = lambda s: bio.Bioinformatics.is_a_valid_smile_or_psmile(s)


import pytest
@pytest.mark.above10s
def test_():
    # pixi run pytest -rFP -q -s bio\combined_generate.py -o "addopts="
    config = CombinedGenerateConfig()
    config.smiles_to_generate = 10
    config.batch_size = 1
    config.model_dir = CHECKPOINT_TEST_FOLDER
    bio.cacca_generate.run_with_config(config)
    
def main():
    config = tyro.cli(CombinedGenerateConfig)
    bio.cacca_generate.run_with_config(config)
