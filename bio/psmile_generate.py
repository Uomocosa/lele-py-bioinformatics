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
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

CHECKPOINT_FOLDER = lele.P(r"./PSMILES_checkpoints") 
CHECKPOINT_TEST_FOLDER = lele.P(r"./PSMILES_checkpoints_test") 

@dataclass
class PSmileGenerateConfig(bio.cacca_generate.GenerateConfig):
    model_dir: Path = CHECKPOINT_FOLDER
    is_smile_valid: Callable[[Smile], bool] = lambda psmile: bio.Bioinformatics.PSmileMethod.is_psmile_string_valid(psmile)

import pytest
@pytest.mark.above10s
def test_():
    config = PSmileGenerateConfig()
    config.smiles_to_generate = 10
    config.batch_size = 1
    config.model_dir = CHECKPOINT_TEST_FOLDER
    bio.cacca_generate.run_with_config(config)
    
def main():
    config = tyro.cli(PSmileGenerateConfig)
    bio.cacca_generate.run_with_config(config)
