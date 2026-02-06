"""
DEPRECATED:
    use: MinGPT.ModelConfig.save()
"""

import json
from pathlib import Path
from lele.String import P
from lele.Json import save_dict_to_json_file
from loguru import logger

def save_model_config(config: dict, file: Path):
    logger.warning("DEPRECATED: use MinGPT.ModelConfig.load()")
    file_Path = P(file)
    config_ = config.copy()
    device = config_.pop("device")
    proper_config = {device: config_}
    save_dict_to_json_file(proper_config, file_Path)


def test_():
    from bio.MinGPT.__global__ import HELPER_DIR
    config = {
        "device": "cpu",
        "path": P('.'),
        "epochs": 10,
        "hello there!": "general kenobi",
    }
    save_model_config(config, HELPER_DIR/"example_model_config.json")
