"""
DEPRECATED:
    use: MinGPT.ModelConfig.load()
"""

import torch
import lele, bio
from bio.ML import get_torch_device
from loguru import logger

def load_model_config(file_Path):
    logger.warning("DEPRECATED: use MinGPT.ModelConfig.load()")
    file_Path = lele.P(file_Path)
    assert file_Path.exists(), f"'{file_Path}' file not found"
    if file_Path.suffix == ".json":
        import json

        with open(file_Path, "r") as f:
            full_config = json.load(f)
    elif file_Path.suffix == ".jsonc":
        full_config = lele.Json.get_dict_from_jsonc(file_Path)
    else:
        raise FileExistsError(f"File extension '{file_Path.suffix}' not supported")

    DEVICE = get_torch_device()
    CONFIG = dict()
    keys = list(full_config.keys())
    if "common" in keys:
        CONFIG = CONFIG | full_config["common"]
    if DEVICE in keys:
        CONFIG = {"device": DEVICE} | CONFIG | full_config[DEVICE]
    elif len(keys) == 1:
        CONFIG = {"device": DEVICE} | CONFIG | full_config[keys[0]]
    else:
        KeyError(f"Could not understand model_config.json file")
    return CONFIG


def test_():
    from bio.MinGPT.__global__ import HELPER_DIR
    print(load_model_config(HELPER_DIR / "example_model_config.jsonc"))
